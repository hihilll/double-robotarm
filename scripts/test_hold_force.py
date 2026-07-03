"""任务#22: 恒力按压验收（§4.4 测试②③）。

流程: 手动把 UR5 移到桌面上方约 2cm(示教器) → 运行本脚本 →
guarded_move 下压接触 → insert 模式恒力 10N 保持 30s → 统计波动 → 抬起。
期间可人为侧推工件测恢复(测试③)。

用法:  python scripts/test_hold_force.py --force 10 --duration 30
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.setup import admittance_params, build_ur5_stack
from control.admittance import AdmittanceController
from control.task_frame import TaskFrame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", type=float, default=10.0)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--payload", default="none")
    args = ap.parse_args()

    stack = build_ur5_stack(payload=args.payload)
    arm, prim = stack.arm, stack.prim
    try:
        task = TaskFrame(arm.tcp_pose())        # 任务系=当前工具系, +Z=工具轴
        input(f"确认工具正对桌面且距离约 2cm, 回车开始({args.force}N × {args.duration}s)...")
        prim.guarded_move(task, np.array([0, 0, 1.0]), 0.005, f_stop=4.0)

        params = admittance_params(stack.ac, "insert")
        adm = AdmittanceController(params)
        wrench_ref = np.array([0, 0, -args.force, 0, 0, 0])
        ref = arm.tcp_pose()
        fz, t_end = [], time.monotonic() + args.duration
        while time.monotonic() < t_end:
            t0 = arm.init_period()
            w, age = prim.wrench_H(task)
            dpos, drot = adm.step(w, arm.dt, wrench_ref)
            target = task.compose_target(ref, dpos, drot)
            stack.prim.safety.check(w, age, arm.tcp_pose(), target)
            arm.servo_l(target)
            fz.append(-w[2])
            arm.wait_period(t0)
        arm.servo_stop()

        # 抬起 2cm 脱离
        up = arm.tcp_pose()
        up[:3] -= task.R[:, 2] * 0.02
        arm.move_l(up, 0.02)

        fz = np.array(fz[len(fz) // 10:])       # 去掉前 10% 过渡段
        err = np.abs(fz - args.force)
        ok = err.max() < 1.0
        print(f"\n恒力 {args.force}N: 均值 {fz.mean():.2f}N, 最大偏差 {err.max():.2f}N, "
              f"标准差 {fz.std():.3f}N → {'达标 ✓ (M2)' if ok else '未达标 ✗(调 D/gain, 查传感器帧率)'}")
    finally:
        stack.close()


if __name__ == "__main__":
    main()
