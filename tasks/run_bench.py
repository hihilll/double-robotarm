"""阶段3桌面调试入口(#25–28): 治具固定桌面、{H} 手工量出, 只动 UR5 不用 UR10。

{H} 来源(§5.1 #24): 示教针尖触孔口3点量出, 或 CAD。竖直向下的孔:
rx=3.1416(工具Z朝下时 {H}+Z 指向孔内)。

用法:
    python tasks/run_bench.py --task plug_insert --H 0.45 0.02 0.15 3.1416 0 0
    python tasks/run_bench.py --task plug_insert --H ... --repeat 20   # #28 验收
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.logger import RunLogger
from common.setup import build_ur5_stack
from control.task_frame import TaskFrame
from tasks.dual_arm_manager import PAYLOAD_OF
from tasks.insertion import InsertionTask
from tasks.plug_extract import PlugExtractTask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(PAYLOAD_OF), required=True)
    ap.add_argument("--H", nargs=6, type=float, required=True,
                    help="{H} 世界系位姿 x y z rx ry rz")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--perturb-mm", type=float, default=0.0,
                    help="每次给 {H} 加 ±此值(mm)的XY均匀随机扰动(#28 验收用 3.0)")
    args = ap.parse_args()

    stack = build_ur5_stack(payload=PAYLOAD_OF[args.task])
    stack.prim.logger = RunLogger(f"bench_{args.task}")
    rng = np.random.default_rng()
    results = []
    try:
        for i in range(args.repeat):
            H_pose = np.asarray(args.H, float)
            if args.perturb_mm > 0:
                H_pose[:2] += rng.uniform(-1, 1, 2) * args.perturb_mm * 1e-3
            H = TaskFrame(H_pose)
            if args.task == "plug_extract":
                info = PlugExtractTask(stack).run(H)
            else:
                info = InsertionTask(stack, args.task).run(H)
            results.append(info)
            print(f"[{i + 1}/{args.repeat}] {'成功' if info['success'] else '失败@' + str(info['fail_stage'])} "
                  f" 用时 {info['t_total']:.1f}s (搜孔 {info['t_search']:.1f}s, 尝试 {info['attempts']})")
    finally:
        stack.prim.logger.close()
        stack.close()

    n_ok = sum(r["success"] for r in results)
    print(f"\n成功率 {n_ok}/{len(results)} = {n_ok / len(results):.0%}"
          f"  (#28 验收标准: ±3mm 扰动 20 次 ≥90%)")


if __name__ == "__main__":
    main()
