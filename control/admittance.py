"""导纳控制器（§4.1 / §4.2）。

在任务系 {H} 下按通道独立积分:  M·ẍ + D·ẋ + K·x = F_ext − F_ref
每个 125Hz 周期调用一次 step()，返回 {H} 系下的柔顺偏移 (dpos, drot)，
由 TaskFrame.compose_target() 合成 servoL 目标。控制器本身不做 I/O。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AdmittanceParams:
    m: np.ndarray   # (3,) 平动虚拟质量 kg
    d: np.ndarray   # (3,) 平动阻尼 N·s/m
    k: np.ndarray   # (3,) 平动刚度 N/m（k=0 的通道由 wrench_ref 做恒力控制）
    mr: np.ndarray  # (3,) 转动惯量 kg·m²
    dr: np.ndarray  # (3,) 转动阻尼 N·m·s/rad
    kr: np.ndarray  # (3,) 转动刚度 N·m/rad
    max_offset: float = 0.02
    max_vel: float = 0.10
    max_rot_offset: float = 0.26
    max_rot_vel: float = 0.5

    @classmethod
    def from_dict(cls, mode: dict, limits: dict) -> "AdmittanceParams":
        """从 config/admittance.yaml 的 modes.<name> + limits 构造。"""
        arr = lambda key: np.asarray(mode[key], float)
        return cls(m=arr("m"), d=arr("d"), k=arr("k"),
                   mr=arr("mr"), dr=arr("dr"), kr=arr("kr"),
                   max_offset=limits["max_offset"], max_vel=limits["max_vel"],
                   max_rot_offset=limits["max_rot_offset"],
                   max_rot_vel=limits["max_rot_vel"])


class AdmittanceController:
    def __init__(self, params: AdmittanceParams):
        self.p = params
        self.reset()

    def reset(self) -> None:
        self.x = np.zeros(3)    # {H} 系平动偏移
        self.v = np.zeros(3)
        self.rx = np.zeros(3)   # {H} 系转动偏移（小角度轴角）
        self.rv = np.zeros(3)

    def set_params(self, params: AdmittanceParams, keep_state: bool = True) -> None:
        """切换模式（search→insert）时热切参数；keep_state 保持偏移连续避免跳变。"""
        self.p = params
        if not keep_state:
            self.reset()

    def step(self, wrench_h: np.ndarray, dt: float,
             wrench_ref: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """wrench_h: {H} 系外部接触力（已重力补偿+折算到TCP）。
        wrench_ref: 期望接触力（如搜孔时 [0,0,-f_push,0,0,0]），k=0 通道靠它推进。"""
        e = wrench_h - (wrench_ref if wrench_ref is not None else 0.0)
        p = self.p

        acc = (e[:3] - p.d * self.v - p.k * self.x) / p.m
        self.v = np.clip(self.v + acc * dt, -p.max_vel, p.max_vel)
        self.x = np.clip(self.x + self.v * dt, -p.max_offset, p.max_offset)

        racc = (e[3:6] - p.dr * self.rv - p.kr * self.rx) / p.mr
        self.rv = np.clip(self.rv + racc * dt, -p.max_rot_vel, p.max_rot_vel)
        self.rx = np.clip(self.rx + self.rv * dt, -p.max_rot_offset, p.max_rot_offset)

        return self.x.copy(), self.rx.copy()
