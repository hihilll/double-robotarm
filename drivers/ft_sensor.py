"""六维力传感器驱动：采集线程 + 低通滤波 + 重力补偿。

默认实现 ATI NetFT 的 RDT/UDP 协议(Axia80 同款)。国产坤维/宇立换成对应 SDK 时，
只需保证 FTReader.latest() 的接口不变：返回传感器系下的 wrench [Fx,Fy,Fz,Tx,Ty,Tz]。
"""
from __future__ import annotations

import math
import socket
import struct
import threading
from typing import Optional, Tuple

import numpy as np

from drivers.bus import LatestValue

G = np.array([0.0, 0.0, -9.81])  # 世界系重力加速度


class FTReader:
    """后台线程持续收流，主控环通过 latest() 取最新滤波值（不排队）。"""

    _RDT_REQUEST = struct.Struct(">HHI")   # (0x1234, command, sample_count)
    _RDT_RECORD = struct.Struct(">IIIiiiiii")  # seq, ft_seq, status, Fx..Tz (counts)

    def __init__(self, ip: str, port: int = 49152,
                 counts_per_force: float = 1e6, counts_per_torque: float = 1e6,
                 lpf_cutoff_hz: float = 15.0):
        self.addr = (ip, port)
        self.cpf = counts_per_force
        self.cpt = counts_per_torque
        self.lpf_cutoff = lpf_cutoff_hz
        self._buf = LatestValue()
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._filt = np.zeros(6)
        self._alpha: Optional[float] = None  # 一阶低通系数, 按实际到帧间隔计算

    # ------------------------------------------------------------ 生命周期
    def start(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(0.5)
        # command=0x0002: 开始高速实时流; sample_count=0: 无限
        self._sock.sendto(self._RDT_REQUEST.pack(0x1234, 0x0002, 0), self.addr)
        self._running = True
        self._thread = threading.Thread(target=self._rx_loop, daemon=True,
                                        name="ft_rx")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._sock is not None:
            try:
                self._sock.sendto(self._RDT_REQUEST.pack(0x1234, 0x0000, 0), self.addr)
            except OSError:
                pass
            self._sock.close()
            self._sock = None

    # ------------------------------------------------------------ 采集线程
    def _rx_loop(self) -> None:
        import time
        last_t = None
        while self._running:
            try:
                data, _ = self._sock.recvfrom(64)
            except socket.timeout:
                continue  # 超时由消费端的 SafetyMonitor 按数据年龄判定
            _, _, _, fx, fy, fz, tx, ty, tz = self._RDT_RECORD.unpack_from(data)
            raw = np.array([fx / self.cpf, fy / self.cpf, fz / self.cpf,
                            tx / self.cpt, ty / self.cpt, tz / self.cpt])
            now = time.monotonic()
            if last_t is not None:
                dt = now - last_t
                self._alpha = 1.0 - math.exp(-2 * math.pi * self.lpf_cutoff * dt)
                self._filt += self._alpha * (raw - self._filt)
            else:
                self._filt = raw
            last_t = now
            self._buf.put(self._filt.copy())

    # ------------------------------------------------------------ 消费接口
    def latest(self) -> Tuple[Optional[np.ndarray], float]:
        """返回 (传感器系 wrench, 数据年龄秒)。安全层用年龄做看门狗。"""
        return self._buf.get()


class GravityCompensator:
    """负载重力补偿(§3.2)。参数来自 calib/payload_id.py，存于 robots.yaml payloads。

    约定: mass(kg)、cog(负载质心, 传感器系, m)、bias(六通道零偏, 传感器单位)。
    """

    def __init__(self, mass: float, cog: np.ndarray, bias: np.ndarray,
                 R_tcp_sensor: np.ndarray):
        self.mass = mass
        self.cog = np.asarray(cog, float)
        self.bias = np.asarray(bias, float)
        self.R_tcp_sensor = np.asarray(R_tcp_sensor, float)

    def compensate(self, wrench_raw: np.ndarray, R_base_tcp: np.ndarray) -> np.ndarray:
        """输入原始 wrench(传感器系)和当前 TCP 姿态，输出纯外部接触 wrench(传感器系)。"""
        R_base_sensor = R_base_tcp @ self.R_tcp_sensor
        g_s = R_base_sensor.T @ G                 # 重力方向在传感器系的投影
        f_g = self.mass * g_s
        t_g = np.cross(self.cog, f_g)
        return wrench_raw - self.bias - np.concatenate([f_g, t_g])
