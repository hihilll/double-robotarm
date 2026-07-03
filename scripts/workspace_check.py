"""任务#15: UR5 呈现区可达性/奇异位形扫描（§7.3）。

在给定中心周围做三维网格 × 多个插入倾角采样，逐点请求控制器解 IK，
统计不可达与腕部奇异(|sin(q5)|过小)的比例。用结果定稿 UR10 present_pose。

说明: getInverseKinematics 由 UR 控制器计算, 需连接真机(不动机器人)。

用法:
    python scripts/workspace_check.py --center 0.45 0.0 0.25 --size 0.30 --step 0.05
"""
from __future__ import annotations

import argparse
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drivers.ur_arm import URArm

SING_MARGIN = math.sin(math.radians(15))   # |sin(q5)| 低于此视为接近腕部奇异


def entry_orientation(tilt_deg: float, azim_deg: float) -> np.ndarray:
    """工具 Z 朝下(-Z 基座) 再绕水平轴倾斜 tilt 的姿态, 返回 rotvec。"""
    R = (Rot.from_euler("z", azim_deg, degrees=True)
         * Rot.from_euler("y", tilt_deg, degrees=True)
         * Rot.from_euler("x", 180, degrees=True))
    return R.as_rotvec()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", nargs=3, type=float, required=True, help="呈现区中心(UR5基座系, m)")
    ap.add_argument("--size", type=float, default=0.30, help="立方体边长 m")
    ap.add_argument("--step", type=float, default=0.05, help="网格间距 m")
    ap.add_argument("--tilts", nargs="*", type=float, default=[0, 15, 30], help="插入倾角 deg")
    args = ap.parse_args()

    cfg = yaml.safe_load((Path(__file__).parents[1] / "config" / "robots.yaml").read_text(encoding="utf-8"))
    arm = URArm("ur5", cfg["ur5"]["ip"], cfg["ur5"]["frequency"], cfg["ur5"]["tcp_offset"])
    arm.connect(control=True)   # 只解 IK, 不发运动指令
    q_near = np.radians(cfg["ur5"]["home_j_deg"]).tolist()

    half = args.size / 2
    grid = np.arange(-half, half + 1e-9, args.step)
    center = np.asarray(args.center, float)
    n_total = n_unreach = n_sing = 0
    bad: list[tuple] = []

    try:
        for dx, dy, dz in product(grid, grid, grid):
            p = center + [dx, dy, dz]
            for tilt in args.tilts:
                for azim in (0, 90, 180, 270):
                    n_total += 1
                    pose = np.concatenate([p, entry_orientation(tilt, azim)])
                    try:
                        q = arm.ctrl.getInverseKinematics(pose.tolist(), q_near)
                    except Exception:
                        q = []
                    if not q:
                        n_unreach += 1
                        bad.append((*np.round(p, 3), tilt, azim, "无解"))
                    elif abs(math.sin(q[4])) < SING_MARGIN:
                        n_sing += 1
                        bad.append((*np.round(p, 3), tilt, azim, f"q5={math.degrees(q[4]):.0f}°近奇异"))
    finally:
        arm.disconnect()

    ok = n_total - n_unreach - n_sing
    print(f"\n采样 {n_total} 个位形: 可用 {ok} ({ok / n_total:.0%}), "
          f"不可达 {n_unreach}, 近奇异 {n_sing}")
    if bad:
        print("问题位形(前20条):")
        for row in bad[:20]:
            print("  ", row)
    print("\n判定: 呈现区应做到 100% 可用; 否则移动中心/缩小区域/调整两臂布置后重扫。(任务#15)")


if __name__ == "__main__":
    main()
