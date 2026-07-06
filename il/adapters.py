"""方案一代码的只读适配层——il 包与方案一之间的唯一接口面。

方案一处于真机验收冻结状态：本包对其只 import、绝不修改。
方案一接口若在真机调试中变化，只需要改这一个文件。
依赖的方案一符号（全部只读）:
    drivers.ur_arm.URArm / drivers.ft_sensor.FTReader, GravityCompensator
    control.admittance.AdmittanceController, AdmittanceParams
    control.safety.SafetyMonitor / control.task_frame.TaskFrame
    common.setup.build_ur5_stack, build_ur10, build_gripper, load_configs
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.setup import (UR5Stack, build_gripper, build_ur5_stack,  # noqa: F401,E402
                          build_ur10, load_configs)
from control.admittance import AdmittanceParams  # noqa: E402

IL_DIR = ROOT / "il"


def load_il_config() -> dict:
    return yaml.safe_load((IL_DIR / "config_il.yaml").read_text(encoding="utf-8"))


def params_from_stiffness(k, kr, il_cfg: dict, limits: dict | None = None) -> AdmittanceParams:
    """由刚度(策略输出/遥操作档位)派生完整导纳参数: d = max(d_min, 2ζ√(k·m))。

    limits 默认取方案一 admittance.yaml 的 limits(只读), 与 M2 验收一致。
    """
    a = il_cfg["admittance"]
    k = np.asarray(k, float)
    kr = np.asarray(kr, float)
    m, mr = np.asarray(a["m"], float), np.asarray(a["mr"], float)
    zeta = float(a["zeta"])
    d = np.maximum(np.asarray(a["d_min"], float), 2.0 * zeta * np.sqrt(np.maximum(k, 0.0) * m))
    dr = np.maximum(np.asarray(a["dr_min"], float), 2.0 * zeta * np.sqrt(np.maximum(kr, 0.0) * mr))
    if limits is None:
        _, ac = load_configs()
        limits = ac["limits"]
    return AdmittanceParams(m=m, d=d, k=k, mr=mr, dr=dr, kr=kr,
                            max_offset=limits["max_offset"], max_vel=limits["max_vel"],
                            max_rot_offset=limits["max_rot_offset"],
                            max_rot_vel=limits["max_rot_vel"])


def stiffness_of_mode(il_cfg: dict, mode: str) -> tuple[np.ndarray, np.ndarray]:
    m = il_cfg["stiffness_modes"][mode]
    return np.asarray(m["k"], float), np.asarray(m["kr"], float)


MODE_CYCLE = ("low", "mid", "high")


def clamp_stiffness(k, kr, il_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """部署期对策略输出刚度做硬限幅(deploy.stiffness_clamp)。"""
    c = il_cfg["deploy"]["stiffness_clamp"]
    return (np.clip(np.asarray(k, float), c["k"][0], c["k"][1]),
            np.clip(np.asarray(kr, float), c["kr"][0], c["kr"][1]))
