"""任务#14: 两臂低速点动测试（阶段0验收第二步）。

沿基座系 X/Y/Z 各 ±2cm 往返一次；UR10 可加 --present 测试呈现位姿往返。
⚠ 示教器速度滑块 30%，手扶急停；先空载运行。

用法:
    python scripts/jog_test.py --arm ur5
    python scripts/jog_test.py --arm ur10 --present
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drivers.ur_arm import URArm

JOG = 0.02   # m
V, A = 0.05, 0.3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["ur5", "ur10"], required=True)
    ap.add_argument("--present", action="store_true", help="UR10: 测试 present_pose 往返")
    args = ap.parse_args()

    cfg = yaml.safe_load((Path(__file__).parents[1] / "config" / "robots.yaml").read_text(encoding="utf-8"))
    c = cfg[args.arm]
    arm = URArm(args.arm, c["ip"], c["frequency"])
    arm.connect(control=True)
    try:
        home = arm.tcp_pose()
        print(f"[{args.arm}] 当前 TCP: {np.round(home, 4)}")
        input("确认末端周围 5cm 内无障碍, 回车开始点动...")

        for axis, name in [(0, "X"), (1, "Y"), (2, "Z")]:
            for sign in (+1, -1):
                target = home.copy()
                target[axis] += sign * JOG
                print(f"  {name}{'+' if sign > 0 else '-'}{JOG * 1000:.0f}mm ...", end=" ")
                arm.move_l(target, V, A)
                arm.move_l(home, V, A)
                print("OK")

        if args.present and args.arm == "ur10":
            pp = np.asarray(c["present_pose"], float)
            print(f"呈现位姿往返: {np.round(pp, 4)}")
            input("确认路径无障碍, 回车继续...")
            arm.move_l(pp, 0.1, 0.3)
            input("已到呈现位姿, 观察姿态是否合理, 回车返回...")
            arm.move_l(home, 0.1, 0.3)

        print(f"\n[{args.arm}] 点动测试通过 ✓ (任务#14)")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
