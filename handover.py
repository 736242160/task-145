#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
handover.py — 基站间切换裁决工具（纯 Python 标准库，单文件）

输入格式（文本行，# 之后为注释，空行忽略）：
    BS <名称> <容量> <x1> <y1> <x2> <y2> <信道号>   基站定义，覆盖范围为矩形
    EV <用户> <x> <y> <基站:信号dBm> [...]          用户移动/测量事件

切换决策规则（自定）：
    1. 用户无服务基站 -> 初始接入信号最强的候选基站。
    2. 服务基站信号 < -90dBm（变弱）或当前位置已离开其覆盖范围 -> 必须切换。
    3. 否则仅当最强候选信号超过服务信号 5dB（迟滞余量）才切换，避免来回抖动。
    4. 候选按信号强度降序尝试；某候选校验失败则记录报告并回退尝试次优候选。

校验与报告：
    - 目标基站不存在            -> 错误
    - 目标覆盖范围不含用户位置  -> 错误
    - 目标容量已满（超容量）    -> 错误（报告用户与基站）
    - 源/目标同频或邻频且覆盖重叠 -> 警告（干扰规则：信道号之差 <= 1 且矩形相交）
    - 同一用户在同一对基站间累计切换 >= 3 次 -> 警告（乒乓，之后每多 2 次再报）

状态跨事件延续：用户服务关系、各基站负载、切换历史均在事件间保留。

用法：
    python3 handover.py 输入文件
    python3 handover.py < 输入文件
    python3 handover.py --demo        运行内置自测样例
"""

import argparse
import sys
from dataclasses import dataclass

WEAK_THRESHOLD = -90.0   # dBm，服务信号低于该值视为变弱
HYSTERESIS_DB = 5.0      # 候选须超过服务信号的迟滞余量
PINGPONG_MIN = 3         # 同一对基站间累计切换达到该次数判定乒乓
ADJACENT_CHANNEL = 1     # 信道号之差 <= 该值视为同频/邻频


@dataclass
class BaseStation:
    name: str
    capacity: int
    x1: float
    y1: float
    x2: float
    y2: float
    freq: int
    load: int = 0

    def covers(self, x, y):
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def overlaps(self, other):
        return not (self.x2 < other.x1 or other.x2 < self.x1
                    or self.y2 < other.y1 or other.y2 < self.y1)


@dataclass
class Report:
    event_no: int
    severity: str  # 错误 / 警告
    kind: str
    user: str
    detail: str


def fmt_sig(value):
    return "丢失" if value is None else f"{value:g}dBm"


class Engine:
    def __init__(self):
        self.bs = {}          # 名称 -> BaseStation
        self.serving = {}     # 用户 -> 基站名称
        self.pair_count = {}  # 用户 -> {frozenset({站A,站B}): 次数}
        self.reports = []
        self.out = []

    def add_bs(self, bs):
        if bs.name in self.bs:
            raise ValueError(f"基站重复定义: {bs.name}")
        self.bs[bs.name] = bs

    def report(self, event_no, severity, kind, user, detail):
        self.reports.append(Report(event_no, severity, kind, user, detail))
        mark = "!!" if severity == "错误" else " !"
        self.out.append(f"  {mark} [{severity}][{kind}] 用户 {user} | {detail}")

    def do_event(self, event_no, user, x, y, signals):
        sig_desc = ", ".join(f"{n}:{s:g}" for n, s in signals.items()) or "无"
        self.out.append(f"[事件{event_no}] 用户 {user} @ ({x:g},{y:g}) 信号[{sig_desc}]")

        cur_name = self.serving.get(user)
        cur = self.bs.get(cur_name) if cur_name else None
        cur_sig = signals.get(cur_name) if cur_name else None
        cands = sorted(((n, s) for n, s in signals.items() if n != cur_name),
                       key=lambda kv: kv[1], reverse=True)

        if cur is None:
            self.out.append("  决策: 用户无服务基站，执行初始接入")
            self._try_candidates(event_no, user, x, y, None, cands)
            return

        left_cover = not cur.covers(x, y)
        weak = cur_sig is None or cur_sig < WEAK_THRESHOLD
        if left_cover or weak:
            reasons = []
            if left_cover:
                reasons.append(f"位置已离开 {cur_name} 覆盖范围")
            if weak:
                reasons.append(f"服务信号 {fmt_sig(cur_sig)} 低于阈值 {WEAK_THRESHOLD:g}dBm")
            self.out.append(f"  决策: {'；'.join(reasons)}，触发切换")
            self._try_candidates(event_no, user, x, y, cur, cands)
            return

        if cands and cands[0][1] > cur_sig + HYSTERESIS_DB:
            self.out.append(
                f"  决策: 候选 {cands[0][0]} 信号 {cands[0][1]:g}dBm 超过服务信号 "
                f"{cur_sig:g}dBm + 迟滞 {HYSTERESIS_DB:g}dB，触发切换")
            self._try_candidates(event_no, user, x, y, cur, cands)
        else:
            self.out.append(f"  决策: 保持连接 {cur_name}（信号 {cur_sig:g}dBm，无需切换）")

    def _try_candidates(self, event_no, user, x, y, source, cands):
        action = "切换" if source else "接入"
        if not cands:
            self.out.append(f"  => {action}失败: 无候选基站"
                            + (f"，用户保持连接 {source.name}" if source else ""))
            return
        for name, sig in cands:
            target = self.bs.get(name)
            if target is None:
                self.report(event_no, "错误", "未知基站", user,
                            f"切换目标 {name} 不存在，尝试次优候选")
                continue
            if not target.covers(x, y):
                self.report(event_no, "错误", "覆盖越界", user,
                            f"目标 {target.name} 覆盖范围不含位置 ({x:g},{y:g})，尝试次优候选")
                continue
            if target.load >= target.capacity:
                self.report(event_no, "错误", "容量已满", user,
                            f"目标 {target.name} 容量 {target.load}/{target.capacity} 已满，"
                            f"切换失败，尝试次优候选")
                continue
            if source is not None and self._interferes(source, target):
                diff = abs(source.freq - target.freq)
                kind = "同频" if diff == 0 else "邻频"
                self.report(event_no, "警告", "基站干扰", user,
                            f"源 {source.name}(信道{source.freq}) 与目标 "
                            f"{target.name}(信道{target.freq}) {kind}且覆盖重叠")
            if source is not None:
                source.load -= 1
            target.load += 1
            self.serving[user] = target.name
            if source is not None:
                self.out.append(
                    f"  => 切换成功: {source.name} -> {target.name}（信号 {sig:g}dBm）")
                self._check_pingpong(event_no, user, source.name, target.name)
            else:
                self.out.append(f"  => 初始接入: {target.name}（信号 {sig:g}dBm）")
            return
        self.out.append(f"  => {action}失败: 所有候选均不可用"
                        + (f"，用户保持连接 {source.name}" if source else "，用户无服务"))

    def _interferes(self, a, b):
        return abs(a.freq - b.freq) <= ADJACENT_CHANNEL and a.overlaps(b)

    def _check_pingpong(self, event_no, user, a, b):
        key = frozenset((a, b))
        counts = self.pair_count.setdefault(user, {})
        counts[key] = counts.get(key, 0) + 1
        n = counts[key]
        if n >= PINGPONG_MIN and (n - PINGPONG_MIN) % 2 == 0:
            self.report(event_no, "警告", "乒乓切换", user,
                        f"用户 {user} 已在 {a} 与 {b} 之间反复切换 {n} 次")

    def summary(self):
        lines = ["", "========== 错误/警告报告清单 =========="]
        if not self.reports:
            lines.append("（无）")
        for r in self.reports:
            lines.append(f"[{r.severity}][{r.kind}] 事件{r.event_no} | 用户 {r.user} | {r.detail}")
        errs = sum(1 for r in self.reports if r.severity == "错误")
        warns = len(self.reports) - errs
        lines.append(f"合计: {errs} 个错误, {warns} 个警告")
        lines.append("")
        lines.append("========== 最终状态 ==========")
        for bs in self.bs.values():
            lines.append(f"基站 {bs.name}: 负载 {bs.load}/{bs.capacity} (信道 {bs.freq})")
        for user, name in sorted(self.serving.items()):
            lines.append(f"用户 {user}: 服务基站 {name}")
        return lines


def parse(text):
    bs_defs, events = [], []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tok = line.split()
        try:
            if tok[0] == "BS":
                if len(tok) != 8:
                    raise ValueError("BS 需要 7 个参数: 名称 容量 x1 y1 x2 y2 信道")
                bs_defs.append(BaseStation(tok[1], int(tok[2]), float(tok[3]),
                                           float(tok[4]), float(tok[5]),
                                           float(tok[6]), int(tok[7])))
            elif tok[0] == "EV":
                if len(tok) < 4:
                    raise ValueError("EV 至少需要: 用户 x y [基站:信号 ...]")
                signals = {}
                for item in tok[4:]:
                    name, _, val = item.rpartition(":")
                    if not name or not val:
                        raise ValueError(f"信号项格式应为 基站:信号，得到 {item!r}")
                    signals[name] = float(val)
                events.append((tok[1], float(tok[2]), float(tok[3]), signals))
            else:
                raise ValueError(f"未知指令 {tok[0]!r}（应为 BS 或 EV）")
        except ValueError as exc:
            raise ValueError(f"第 {lineno} 行: {exc} | {raw.strip()}") from None
    return bs_defs, events


DEMO = """\
# 基站定义: BS 名称 容量 x1 y1 x2 y2 信道
BS A 3 0 0 100 100 1
BS B 2 80 0 180 100 2
BS C 3 200 0 300 100 5
BS D 1 90 90 190 190 1

# 初始接入
EV u1 10 10 A:-60
# u1 信号变弱 -> A->B，A(信道1)与B(信道2)邻频且覆盖重叠 -> 干扰警告
EV u1 90 50 A:-96 B:-70
EV u2 20 20 A:-65
EV u3 30 30 A:-70
EV u4 40 40 A:-70
# A 已满(3/3) -> u5 接入 A 失败(容量)，回退接入 B
EV u5 90 50 A:-55 B:-80
# u2 变弱 -> 目标 B 已满 -> 容量错误，保持 A
EV u2 90 50 A:-95 B:-65
# u1 离开 B 覆盖 -> B->C
EV u1 250 50 C:-60
# u4 离开 A 覆盖 -> A->D，A 与 D 同频(信道1)且重叠 -> 干扰警告
EV u4 150 150 D:-60
# u6 在 A/B 间反复切换 -> 乒乓警告
EV u6 90 50 A:-70 B:-75
EV u6 90 50 A:-80 B:-65
EV u6 90 50 A:-65 B:-80
EV u6 90 50 A:-80 B:-65
# 引用不存在的基站 X -> 错误
EV u7 250 50 X:-50
EV u7 250 50 C:-60
# 最强候选 D 不含该位置 -> 覆盖越界错误，回退接入 C
EV u8 250 50 D:-50 C:-75
"""


def run(text):
    bs_defs, events = parse(text)
    eng = Engine()
    for bs in bs_defs:
        eng.add_bs(bs)
    for i, (user, x, y, signals) in enumerate(events, 1):
        eng.do_event(i, user, x, y, signals)
    print("\n".join(eng.out + eng.summary()))
    return sum(1 for r in eng.reports if r.severity == "错误")


def main(argv=None):
    ap = argparse.ArgumentParser(description="基站间切换裁决工具（纯标准库）")
    ap.add_argument("input", nargs="?", help="输入文件（缺省读标准输入）")
    ap.add_argument("--demo", action="store_true", help="运行内置自测样例")
    args = ap.parse_args(argv)
    if args.demo:
        text = DEMO
    elif args.input:
        with open(args.input, encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()
    try:
        run(text)
    except ValueError as exc:
        print(f"输入解析失败: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
