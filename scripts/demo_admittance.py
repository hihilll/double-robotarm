"""阶段2验收脚本：手推柔顺演示（§4.4 验收测试①）。

冻结当前 TCP 位姿为参考, 以工具系为任务系跑导纳环——手推末端应顺滑跟随、
松手缓慢回中(hand_push_demo 模式 k=0 则停在原地)。Ctrl+C 退出。

⚠ 首次运行: 示教器速度滑块 30%, 手放在急停上, 确认 robots.yaml 的负载参数已辨识。
用法:  python scripts/demo_admittance.py [--payload none]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.admittance import AdmittanceController, AdmittanceParams
from control.safety import SafetyLimits, SafetyMonitor
from control.task_frame import TaskFrame
from control.primitives import Primitives
from drivers.ft_sensor import FTReader, GravityCompensator
from drivers.ur_arm import URArm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", default="none", help="robots.yaml payloads 键名")
    args = ap.parse_args()

    root = Path(__file__).parents[1]
    rc = yaml.safe_load((root / "config" / "robots.yaml").read_text(encoding="utf-8"))
    ac = yaml.safe_load((root / "config" / "admittance.yaml").read_text(encoding="utf-8"))

    # --- 组装 ---
    arm = URArm("ur5", rc["ur5"]["ip"], rc["ur5"]["frequency"], rc["ur5"]["tcp_offset"])
    arm.connect(control=True)

    fc = rc["ft_sensor"]
    ft = FTReader(fc["ip"], fc["port"], fc["counts_per_force"],
                  fc["counts_per_torque"], fc["lpf_cutoff_hz"])
    ft.start()

    pl = rc["payloads"][args.payload]
    gravity = GravityCompensator(pl["mass"], np.array(pl["cog"]),
                                 np.array(pl["bias"]), np.array(fc["R_tcp_sensor"]))
    safety = SafetyMonitor(SafetyLimits.from_dict(ac["safety"]))
    prim = Primitives(arm, ft, gravity, safety,
                      np.array(fc["R_tcp_sensor"]), np.array(fc["p_sensor_to_tcp"]))

    params = AdmittanceParams.from_dict(ac["modes"]["hand_push_demo"], ac["limits"])
    adm = AdmittanceController(params)

    ref = arm.tcp_pose()
    task = TaskFrame(ref)          # 任务系 = 当前工具位姿(轴向即工具轴)
    print("导纳环运行中, 手推末端测试柔顺性; Ctrl+C 退出。")
    try:
        while True:
            t0 = arm.init_period()
            w, age = prim.wrench_H(task)
            dpos, drot = adm.step(w, arm.dt)
            target = task.compose_target(ref, dpos, drot)
            safety.check(w, age, arm.tcp_pose(), target)
            arm.servo_l(target)
            arm.wait_period(t0)
    except KeyboardInterrupt:
        print("\n退出")
    finally:
        arm.servo_stop()
        ft.stop()
        arm.disconnect()


if __name__ == "__main__":
    main()
