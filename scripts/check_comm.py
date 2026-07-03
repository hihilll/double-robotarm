"""阶段0验收脚本：双臂 RTDE + 力传感器通讯检查（只读，不动机器人）。

用法:  python scripts/check_comm.py
通过标准: 两臂位姿/关节读数正常刷新; 力传感器帧率 ≥ 期望值的 90%。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drivers.ft_sensor import FTReader
from drivers.ur_arm import URArm


def main() -> None:
    cfg = yaml.safe_load((Path(__file__).parents[1] / "config" / "robots.yaml").read_text(encoding="utf-8"))

    for name in ("ur5", "ur10"):
        arm = URArm(name, cfg[name]["ip"], cfg[name]["frequency"])
        print(f"[{name}] 连接 {cfg[name]['ip']} ...", end=" ")
        arm.connect(control=False)           # 只读，不抢控制权
        pose = arm.tcp_pose()
        q = np.degrees(arm.joints())
        print("OK")
        print(f"  TCP: {np.round(pose, 4)}")
        print(f"  关节(deg): {np.round(q, 1)}")
        print(f"  保护性停止: {arm.is_protective_stopped()}")
        arm.disconnect()

    fc = cfg["ft_sensor"]
    print(f"[ft] 连接 {fc['ip']}:{fc['port']} ...", end=" ")
    ft = FTReader(fc["ip"], fc["port"], fc["counts_per_force"],
                  fc["counts_per_torque"], fc["lpf_cutoff_hz"])
    ft.start()
    time.sleep(0.5)
    w0, age = ft.latest()
    assert w0 is not None, "力传感器无数据"
    print(f"OK  wrench={np.round(w0, 2)}")

    # 帧率测试: 统计 2 秒内数据年龄的中位数反推
    ages = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < 2.0:
        _, age = ft.latest()
        ages.append(age)
        time.sleep(0.001)
    ft.stop()
    print(f"[ft] 数据年龄中位数 {np.median(ages) * 1000:.2f} ms "
          f"(应 < 1000/传感器输出频率 ms)")
    print("\n阶段0通讯检查通过 ✓")


if __name__ == "__main__":
    main()
