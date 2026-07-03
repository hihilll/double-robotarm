"""安全层（§4.4）：独立于任务逻辑，力控主环每周期调用一次 check()。

任何违规抛 SafetyError；调用方（原语层）捕获后执行 servo_stop + 抬升退出。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class SafetyError(RuntimeError):
    pass


@dataclass
class SafetyLimits:
    f_max: float = 40.0
    t_max: float = 5.0
    ft_timeout: float = 0.05        # s
    max_servo_jump: float = 0.005   # m
    workspace_min: np.ndarray = None
    workspace_max: np.ndarray = None

    @classmethod
    def from_dict(cls, cfg: dict) -> "SafetyLimits":
        return cls(f_max=cfg["f_max"], t_max=cfg["t_max"],
                   ft_timeout=cfg["ft_timeout"], max_servo_jump=cfg["max_servo_jump"],
                   workspace_min=np.asarray(cfg["workspace_min"], float),
                   workspace_max=np.asarray(cfg["workspace_max"], float))


class SafetyMonitor:
    def __init__(self, limits: SafetyLimits):
        self.lim = limits

    def check(self, wrench_ext: np.ndarray, ft_age: float,
              tcp_pose: np.ndarray, target_pose: np.ndarray) -> None:
        lim = self.lim
        f, t = np.linalg.norm(wrench_ext[:3]), np.linalg.norm(wrench_ext[3:6])
        if f > lim.f_max or t > lim.t_max:
            raise SafetyError(f"接触力超限: |F|={f:.1f}N |T|={t:.2f}Nm")
        if ft_age > lim.ft_timeout:
            raise SafetyError(f"力传感器数据超时: {ft_age * 1000:.0f}ms")
        if np.linalg.norm(target_pose[:3] - tcp_pose[:3]) > lim.max_servo_jump:
            raise SafetyError("servoL 目标突跳(疑似飞车)")
        p = target_pose[:3]
        if np.any(p < lim.workspace_min) or np.any(p > lim.workspace_max):
            raise SafetyError(f"目标超出工作空间盒: {p}")
