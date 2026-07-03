"""UR CB3 机械臂驱动：ur_rtde 的薄封装，UR5/UR10 共用。

位姿约定与 UR 一致：[x, y, z, rx, ry, rz]，旋转为轴角向量(rotation vector)，单位 m/rad。
一台臂 = 一个 URArm 实例 = 一组独立的 RTDE 连接(30004) + Dashboard(29999)。
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

try:
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
    from dashboard_client import DashboardClient
except ImportError:  # 允许在没装 ur_rtde 的开发机上导入本模块(如跑单元测试)
    RTDEControlInterface = RTDEReceiveInterface = DashboardClient = None  # type: ignore


class URArm:
    def __init__(self, name: str, ip: str, frequency: float = 125.0,
                 tcp_offset: Optional[Sequence[float]] = None):
        self.name = name
        self.ip = ip
        self.frequency = frequency
        self.dt = 1.0 / frequency
        self._tcp_offset = list(tcp_offset) if tcp_offset is not None else None
        self._ctrl: Optional["RTDEControlInterface"] = None
        self._recv: Optional["RTDEReceiveInterface"] = None
        self._dash: Optional["DashboardClient"] = None

    # ------------------------------------------------------------ 连接管理
    def connect(self, control: bool = True) -> None:
        """control=False 时只建接收连接（check_comm 等只读场景，不抢占控制权）。"""
        if RTDEReceiveInterface is None:
            raise RuntimeError("未安装 ur_rtde: pip install ur-rtde")
        self._recv = RTDEReceiveInterface(self.ip, self.frequency)
        if control:
            self._ctrl = RTDEControlInterface(self.ip, self.frequency)
            if self._tcp_offset is not None:
                self._ctrl.setTcp(self._tcp_offset)

    def disconnect(self) -> None:
        if self._ctrl is not None:
            self._ctrl.servoStop()
            self._ctrl.stopScript()
            self._ctrl.disconnect()
            self._ctrl = None
        if self._recv is not None:
            self._recv.disconnect()
            self._recv = None

    @property
    def ctrl(self) -> "RTDEControlInterface":
        assert self._ctrl is not None, f"{self.name}: 控制连接未建立(connect(control=True))"
        return self._ctrl

    @property
    def recv(self) -> "RTDEReceiveInterface":
        assert self._recv is not None, f"{self.name}: 未连接"
        return self._recv

    # ------------------------------------------------------------ 状态读取
    def tcp_pose(self) -> np.ndarray:
        """基座系 TCP 位姿 [x,y,z,rx,ry,rz]。"""
        return np.asarray(self.recv.getActualTCPPose(), dtype=float)

    def tcp_speed(self) -> np.ndarray:
        return np.asarray(self.recv.getActualTCPSpeed(), dtype=float)

    def joints(self) -> np.ndarray:
        return np.asarray(self.recv.getActualQ(), dtype=float)

    def is_protective_stopped(self) -> bool:
        return bool(self.recv.isProtectiveStopped())

    # ------------------------------------------------------------ 运动指令
    def move_l(self, pose: Sequence[float], v: float = 0.1, a: float = 0.5,
               blocking: bool = True) -> None:
        """笛卡尔直线运动（高刚度、非柔顺）。UR10 的所有动作只用这一个。"""
        self.ctrl.moveL(list(pose), v, a, not blocking)

    def move_j_deg(self, q_deg: Sequence[float], v: float = 0.5, a: float = 1.0) -> None:
        self.ctrl.moveJ([math.radians(q) for q in q_deg], v, a)

    def servo_l(self, pose: np.ndarray, lookahead: float = 0.1, gain: float = 300.0) -> None:
        """125Hz 流式笛卡尔伺服——力控主环的唯一下发接口。
        gain 范围 100~2000，越大跟踪越紧但越容易振荡，CB3 上从 300 起调(§8 风险)。"""
        self.ctrl.servoL(pose.tolist(), 0.0, 0.0, self.dt, lookahead, gain)

    def servo_stop(self, decel: float = 2.0) -> None:
        self.ctrl.servoStop(decel)

    def stop_l(self, decel: float = 2.0) -> None:
        self.ctrl.stopL(decel)

    # ------------------------------------------------------------ 节拍同步
    # 用法(力控环体):
    #   t0 = arm.init_period(); ...计算+servo_l...; arm.wait_period(t0)
    def init_period(self):
        return self.ctrl.initPeriod()

    def wait_period(self, t0) -> None:
        self.ctrl.waitPeriod(t0)
