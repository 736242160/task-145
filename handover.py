#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基站间切换裁决工具（纯 Python 标准库，单文件）

用法:
    python3 handover.py 输入文件        # 从文件读取基站定义与事件流
    python3 handover.py                # 从标准输入读取
    python3 handover.py --demo         # 运行内置自测样例

==================== 输入格式 ====================
基站定义（每行一条）:
    BS <名称> <容量> <中心x> <中心y> <覆盖半径>
事件（每行一条，顺序处理，状态跨事件延续）:
    EVT <用户> <位置x> <位置y> <测量1> [测量2 ...]
    测量格式: <基站名>=<信号强度dBm>   例如  A=-70
以 # 开头或空行将被忽略。

==================== 切换决策规则（自定，理由见下） ====================
1. 触发条件（满足其一即尝试切换）:
   a) 当前服务基站信号 < 弱信号门限 WEAK_THRESHOLD(-95 dBm) —— 信号太弱必须走；
   b) 最强候选基站信号 > 当前信号 + 迟滞余量 HYSTERESIS(3 dB) —— 迟滞用于
      抑制边界抖动引起的无谓切换（但不能完全杜绝乒乓，故另有乒乓检测）。
2. 目标选择: 按测量信号从强到弱排序，依次尝试，第一个通过全部校验者胜出。
   校验顺序: 基站存在 -> 覆盖用户位置 -> 容量未满。任一失败记录错误并尝试
   次优候选（容量/覆盖问题是局部问题，次优基站可能可用）。
3. 干扰规则（自定）: 信道按基站定义顺序以复用因子 3 分配（ch = 序号 % 3）。
   源与目标基站满足 同频(信道相同) 或 邻频(信道号相差 1)，且两者覆盖圆
   相交（圆心距 <= 半径之和）时，判定存在干扰，记为告警（不阻断切换，
   因为干扰是性能问题而非硬失败，运维需要知道但用户不应掉线）。
4. 乒乓检测: 用户最近 4 次驻留记录呈 A->B->A->B 交替模式即判定乒乓，
   报告后重置该用户历史以避免重复刷屏。

==================== 错误/告警类别 ====================
- 超容量        : 目标基站容量已满，切换/接入失败（用户、基站）
- 干扰          : 源与目标基站同频/邻频且覆盖重叠（告警，不阻断）
- 乒乓          : 用户在两站间反复切换（两站、用户）
- 不存在的基站  : 事件引用了未定义的基站
- 越界覆盖      : 目标基站覆盖范围不含用户当前位置
- 无可用基站    : 所有候选均不可用，用户无法接入/只能保持原基站
"""

import argparse
import math
import sys
from dataclasses import dataclass, field

WEAK_THRESHOLD = -95.0   # 弱信号门限 (dBm)
HYSTERESIS = 3.0         # 切换迟滞余量 (dB)
NUM_CHANNELS = 3         # 频率复用因子


@dataclass
class BaseStation:
    name: str
    capacity: int
    cx: float
    cy: float
    radius: float
    channel: int = 0
    users: set = field(default_factory=set)

    @property
    def load(self):
        return len(self.users)

    def covers(self, x, y):
        return math.hypot(x - self.cx, y - self.cy) <= self.radius + 1e-9


class HandoverEngine:
    def __init__(self, stations):
        self.bs = {}
        for i, st in enumerate(stations):
            st.channel = i % NUM_CHANNELS
            self.bs[st.name] = st
        self.serving = {}      # user -> 当前基站名（跨事件延续）
        self.history = {}      # user -> 驻留基站序列（用于乒乓检测）
        self.results = []      # 逐事件结果
        self.errors = []       # (类别, 事件号, 描述)

    # ---------- 记录 ----------
    def error(self, kind, evt_no, msg):
        self.errors.append((kind, evt_no, msg))

    def result(self, evt_no, msg):
        self.results.append("[事件 %d] %s" % (evt_no, msg))

    # ---------- 干扰检查 ----------
    def check_interference(self, evt_no, user, src_name, dst_name):
        a, b = self.bs[src_name], self.bs[dst_name]
        dch = abs(a.channel - b.channel)
        if dch > 1:
            return
        kind = "同频" if dch == 0 else "邻频"
        dist = math.hypot(a.cx - b.cx, a.cy - b.cy)
        if dist <= a.radius + b.radius:
            self.error("干扰", evt_no,
                       "用户 %s 切换 %s->%s：源基站 %s(信道%d) 与目标基站 %s(信道%d) "
                       "为%s信道且覆盖区域重叠（圆心距 %.1f <= 半径和 %.1f）"
                       % (user, src_name, dst_name, src_name, a.channel,
                          dst_name, b.channel, kind, dist, a.radius + b.radius))

    # ---------- 乒乓检查 ----------
    def check_pingpong(self, evt_no, user):
        h = self.history[user]
        if len(h) >= 4:
            a, b, c, d = h[-4:]
            if a == c and b == d and a != b:
                self.error("乒乓", evt_no,
                           "用户 %s 在基站 %s 与 %s 之间反复切换（%s->%s->%s->%s）"
                           % (user, a, b, a, b, a, b))
                self.history[user] = [d]  # 重置，避免重复报告

    # ---------- 候选尝试（初始接入与切换共用） ----------
    def try_candidates(self, evt_no, user, x, y, valid, initial):
        current = self.serving.get(user)
        cands = sorted(((s, n) for n, s in valid.items() if n != current),
                       key=lambda t: (-t[0], t[1]))
        if not cands:
            if initial:
                self.error("无可用基站", evt_no,
                           "用户 %s 在 (%.1f,%.1f) 没有任何有效测量，无法接入"
                           % (user, x, y))
            else:
                self.result(evt_no, "用户 %s 触发切换但无候选基站，保持 %s"
                            % (user, current))
            return
        for sig, name in cands:
            bs = self.bs[name]
            if not bs.covers(x, y):
                self.error("越界覆盖", evt_no,
                           "用户 %s 位置 (%.1f,%.1f) 不在目标基站 %s 的覆盖范围内"
                           "（圆心 (%.1f,%.1f) 半径 %.1f），切换到该站被拒绝"
                           % (user, x, y, name, bs.cx, bs.cy, bs.radius))
                continue
            if bs.load >= bs.capacity:
                self.error("超容量", evt_no,
                           "用户 %s 目标基站 %s 容量已满（%d/%d），切换失败"
                           % (user, name, bs.load, bs.capacity))
                continue
            if current is not None:
                self.check_interference(evt_no, user, current, name)
                self.bs[current].users.discard(user)
            bs.users.add(user)
            self.serving[user] = name
            self.history.setdefault(user, []).append(name)
            self.check_pingpong(evt_no, user)
            if initial:
                self.result(evt_no, "用户 %s 初始接入基站 %s（信号 %.1f dBm，负载 %d/%d）"
                            % (user, name, sig, bs.load, bs.capacity))
            else:
                self.result(evt_no, "用户 %s 切换 %s -> %s（目标信号 %.1f dBm，负载 %d/%d）"
                            % (user, current, name, sig, bs.load, bs.capacity))
            return
        if initial:
            self.error("无可用基站", evt_no,
                       "用户 %s 的所有候选基站均不可用（越界或超容量），未能接入" % user)
        else:
            self.result(evt_no, "用户 %s 切换失败（所有候选均不可用），保持基站 %s"
                        % (user, current))

    # ---------- 事件处理 ----------
    def process_event(self, evt_no, user, x, y, measurements):
        valid = {}
        for name, sig in measurements:
            if name not in self.bs:
                self.error("不存在的基站", evt_no,
                           "用户 %s 的信号测量引用了未定义的基站 %s（该测量已忽略）"
                           % (user, name))
                continue
            valid[name] = sig

        current = self.serving.get(user)
        if current is None:
            self.try_candidates(evt_no, user, x, y, valid, initial=True)
            return

        cur_sig = valid.get(current, float("-inf"))  # 未测到当前站视为极弱
        best_sig, best_name = (None, None)
        for name, sig in valid.items():
            if name != current and (best_sig is None or sig > best_sig):
                best_sig, best_name = sig, name

        weak = cur_sig < WEAK_THRESHOLD
        better = best_sig is not None and best_sig > cur_sig + HYSTERESIS
        if not (weak or better):
            cur_txt = "%.1f" % cur_sig if cur_sig != float("-inf") else "未测到"
            best_txt = ("%.1f(%s)" % (best_sig, best_name)) if best_sig is not None else "无"
            self.result(evt_no, "用户 %s 保持基站 %s（当前信号 %s dBm，最优候选 %s，未触发切换）"
                        % (user, current, cur_txt, best_txt))
            return
        reasons = []
        if weak:
            reasons.append("当前信号 %.1f dBm 低于弱信号门限 %.1f dBm"
                           % (cur_sig, WEAK_THRESHOLD))
        if better:
            reasons.append("候选 %s 信号 %.1f dBm 优于当前 %.1f dBm + 迟滞 %.1f dB"
                           % (best_name, best_sig, cur_sig, HYSTERESIS))
        self.result(evt_no, "用户 %s 触发切换：" % user + "；".join(reasons))
        self.try_candidates(evt_no, user, x, y, valid, initial=False)


# ---------- 输入解析 ----------
def parse_input(text):
    stations, events = [], []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "BS":
            if len(parts) != 6:
                raise ValueError("第 %d 行：BS 格式应为: BS 名称 容量 中心x 中心y 半径" % lineno)
            stations.append(BaseStation(parts[1], int(parts[2]),
                                        float(parts[3]), float(parts[4]), float(parts[5])))
        elif parts[0] == "EVT":
            if len(parts) < 4:
                raise ValueError("第 %d 行：EVT 格式应为: EVT 用户 x y [基站=信号 ...]" % lineno)
            user, x, y = parts[1], float(parts[2]), float(parts[3])
            meas = []
            for tok in parts[4:]:
                if "=" not in tok:
                    raise ValueError("第 %d 行：测量格式应为 基站=信号，得到 %r" % (lineno, tok))
                name, sig = tok.split("=", 1)
                meas.append((name, float(sig)))
            events.append((user, x, y, meas))
        else:
            raise ValueError("第 %d 行：未知指令 %r（应为 BS 或 EVT）" % (lineno, parts[0]))
    return stations, events


# ---------- 输出 ----------
def print_report(engine):
    print("=" * 60)
    print("切换决策规则：信号低于 %.1f dBm 必切；候选优于当前 + %.1f dB 迟滞则切；"
          % (WEAK_THRESHOLD, HYSTERESIS))
    print("干扰规则：信道按定义顺序 %% %d 分配，同频或邻频且覆盖圆相交即报告。"
          % NUM_CHANNELS)
    print("=" * 60)
    print("\n---- 基站表 ----")
    for st in engine.bs.values():
        print("基站 %-4s 信道%d 容量%d 覆盖: 圆心(%.1f,%.1f) 半径%.1f 当前负载 %d"
              % (st.name, st.channel, st.capacity, st.cx, st.cy, st.radius, st.load))
    print("\n---- 逐事件切换结果 ----")
    for r in engine.results:
        print(r)
    print("\n---- 错误与告警清单 ----")
    if not engine.errors:
        print("（无）")
    else:
        by_kind = {}
        for kind, no, msg in engine.errors:
            by_kind.setdefault(kind, []).append((no, msg))
        for kind, items in by_kind.items():
            print("【%s】共 %d 条" % (kind, len(items)))
            for no, msg in items:
                print("  [事件 %d] %s" % (no, msg))


# ---------- 内置自测样例 ----------
DEMO_INPUT = """\
# ---- 基站定义: BS 名称 容量 中心x 中心y 半径 ----
BS A 2 0 0 50
BS B 2 60 0 50
BS C 2 30 120 40
BS D 2 30 300 40
# ---- 事件流: EVT 用户 x y 基站=信号 ... ----
EVT u1 20 0 A=-70 B=-85
EVT u2 25 0 A=-68 B=-80
EVT u3 30 0 A=-65 B=-75
EVT u1 50 0 A=-90 B=-70
EVT u1 30 0 A=-70 B=-88
EVT u1 50 0 A=-90 B=-70
EVT u4 30 200 D=-60 A=-100
EVT u5 10 0 A=-70 Z=-60
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description="基站间切换裁决工具（纯标准库）")
    ap.add_argument("input", nargs="?", help="输入文件（缺省读标准输入）")
    ap.add_argument("--demo", action="store_true", help="运行内置自测样例")
    args = ap.parse_args(argv)

    if args.demo:
        text = DEMO_INPUT
        print("（运行内置自测样例）\n输入内容：\n" + text)
    elif args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    stations, events = parse_input(text)
    engine = HandoverEngine(stations)
    for i, (user, x, y, meas) in enumerate(events, 1):
        engine.process_event(i, user, x, y, meas)
    print_report(engine)
    return 0


if __name__ == "__main__":
    sys.exit(main())
