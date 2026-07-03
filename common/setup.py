"""系统装配工厂：从 config 构建各层对象，消除脚本/任务间的样板代码。

用法:
    from common.setup import load_configs, build_ur5_stack, build_ur10, admittance_params
    stack = build_ur5_stack(payload="plug")   # stack.arm / stack.ft / stack.prim
    ...
    stack.close()
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from control.admittance import AdmittanceParams
from control.primitives import Primitives
from control.safety import SafetyLimits, SafetyMonitor
from drivers.ft_sensor import FTReader, GravityCompensator
from drivers.ur_arm import URArm

ROOT = Path(__file__).resolve().parents[1]


def load_configs() -> tuple[dict, dict]:
    rc = yaml.safe_load((ROOT / "config" / "robots.yaml").read_text(encoding="utf-8"))
    ac = yaml.safe_load((ROOT / "config" / "admittance.yaml").read_text(encoding="utf-8"))
    return rc, ac


def admittance_params(ac: dict, mode: str) -> AdmittanceParams:
    return AdmittanceParams.from_dict(ac["modes"][mode], ac["limits"])


@dataclass
class UR5Stack:
    arm: URArm
    ft: FTReader
    prim: Primitives
    rc: dict
    ac: dict

    def close(self) -> None:
        try:
            self.ft.stop()
        finally:
            self.arm.disconnect()


def build_ur5_stack(payload: str = "none", control: bool = True) -> UR5Stack:
    rc, ac = load_configs()
    arm = URArm("ur5", rc["ur5"]["ip"], rc["ur5"]["frequency"], rc["ur5"]["tcp_offset"])
    arm.connect(control=control)

    fc = rc["ft_sensor"]
    ft = FTReader(fc["ip"], fc["port"], fc["counts_per_force"],
                  fc["counts_per_torque"], fc["lpf_cutoff_hz"])
    ft.start()

    pl = rc["payloads"][payload]
    gravity = GravityCompensator(pl["mass"], np.array(pl["cog"]),
                                 np.array(pl["bias"]), np.array(fc["R_tcp_sensor"]))
    safety = SafetyMonitor(SafetyLimits.from_dict(ac["safety"]))
    prim = Primitives(arm, ft, gravity, safety,
                      np.array(fc["R_tcp_sensor"]), np.array(fc["p_sensor_to_tcp"]))
    return UR5Stack(arm=arm, ft=ft, prim=prim, rc=rc, ac=ac)


def build_ur10(control: bool = True) -> tuple[URArm, dict]:
    rc, _ = load_configs()
    arm = URArm("ur10", rc["ur10"]["ip"], rc["ur10"]["frequency"], rc["ur10"]["tcp_offset"])
    arm.connect(control=control)
    return arm, rc


def build_gripper(rc: dict, which: str):
    """构建并初始化夹爪('ur5'/'ur10')。未接线/串口异常时返回 None 并提示,
    任务层对 None 自动降级为手动确认模式(桌面调试可无夹爪)。"""
    from drivers.gripper import DHGripper
    g = rc["grippers"][which]
    try:
        grip = DHGripper(g["port"], g["slave_id"], g["baudrate"])
        grip.connect()
        return grip
    except Exception as e:
        print(f"[setup] {which} 夹爪不可用({e}), 无夹爪模式继续")
        return None
