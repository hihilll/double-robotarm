"""任务#34: 随机化自动评测（§10）。

协议: UR10 呈现位姿叠加 ±3mm/±3° 均匀随机扰动, 每任务 N 次全自动循环;
统计成功率/耗时/尝试次数/失败模式, 输出 markdown 报表到 logs/。

用法:  python scripts/evaluate.py --task plug_insert --n 30 [--vision]
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasks.dual_arm_manager import PAYLOAD_OF, DualArmManager

TARGET = {"plug_insert": 0.90, "plug_extract": 0.95, "screw_place": 0.85}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(PAYLOAD_OF), required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--pos-mm", type=float, default=3.0)
    ap.add_argument("--rot-deg", type=float, default=3.0)
    args = ap.parse_args()

    rng = np.random.default_rng()
    mgr = DualArmManager(args.task, use_vision=args.vision)
    results = []
    try:
        for i in range(args.n):
            perturb = np.concatenate([
                rng.uniform(-1, 1, 3) * args.pos_mm * 1e-3,
                rng.uniform(-1, 1, 3) * np.radians(args.rot_deg)])
            info = mgr.run_episode(present_perturb=perturb)
            results.append(info)
            print(f"[{i + 1}/{args.n}] {'✓' if info['success'] else '✗ ' + str(info['fail_stage'])} "
                  f"{info['t_total']:.1f}s")
    finally:
        mgr.close()

    # ---- 报表 ----
    n_ok = sum(r["success"] for r in results)
    rate = n_ok / len(results)
    times = np.array([r["t_total"] for r in results if r["success"]])
    fails = Counter(r["fail_stage"] for r in results if not r["success"])
    target = TARGET[args.task]
    lines = [
        f"# 评测报告: {args.task}",
        f"- 日期: {time.strftime('%Y-%m-%d %H:%M')}  扰动: ±{args.pos_mm}mm/±{args.rot_deg}°"
        f"  视觉: {args.vision}",
        f"- **成功率: {n_ok}/{len(results)} = {rate:.0%}**  (目标 ≥{target:.0%}"
        f" → {'达标 ✓' if rate >= target else '未达标 ✗'})",
        f"- 成功单次耗时: 均值 {times.mean():.1f}s / 最大 {times.max():.1f}s"
        if len(times) else "- 无成功样本",
        f"- 平均尝试次数: {np.mean([r['attempts'] for r in results]):.2f}",
        f"- 失败模式: {dict(fails) or '无'}",
        "", "复盘指引(§10): 逐条回放失败 episode 的 HDF5 日志 → 定位根因 → 只改"
        " admittance.yaml → 重跑本脚本回归。",
    ]
    report = "\n".join(lines)
    out = Path(__file__).parents[1] / "logs" / f"eval_{args.task}_{time.strftime('%Y%m%d_%H%M%S')}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\n{report}\n\n报表已保存: {out}")


if __name__ == "__main__":
    main()
