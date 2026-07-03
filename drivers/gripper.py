"""大寰(DH-Robotics) PGC/AG 系列夹爪驱动，Modbus-RTU。

寄存器地址以所购型号的《Modbus 通讯手册》为准，下面是 PGC/AG 系列的常见映射，
到货后逐项核对(TODO)。Robotiq 夹爪则换用其 Modbus 映射，接口保持不变。
"""
from __future__ import annotations

import time

try:
    from pymodbus.client import ModbusSerialClient
except ImportError:
    ModbusSerialClient = None  # type: ignore

# --- DH PGC/AG 常见寄存器映射 (TODO: 按手册核对) ---
REG_INIT = 0x0100        # 写 1: 初始化(回零标定)
REG_FORCE = 0x0101       # 力值 20~100 (%)
REG_POSITION = 0x0103    # 目标位置 0~1000 (千分比, 0=全闭)
REG_SPEED = 0x0104       # 速度 1~100 (%)
REG_INIT_STATE = 0x0200  # 1=初始化完成
REG_GRIP_STATE = 0x0201  # 0=运动中 1=到位(空夹) 2=夹住物体 3=物体掉落
REG_POS_NOW = 0x0202     # 当前位置


class DHGripper:
    def __init__(self, port: str, slave_id: int = 1, baudrate: int = 115200):
        if ModbusSerialClient is None:
            raise RuntimeError("未安装 pymodbus: pip install pymodbus")
        self.slave = slave_id
        self.client = ModbusSerialClient(port=port, baudrate=baudrate,
                                         parity="N", stopbits=1, timeout=0.5)

    def connect(self) -> None:
        assert self.client.connect(), "夹爪串口连接失败"
        self.client.write_register(REG_INIT, 1, slave=self.slave)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 10.0:
            r = self.client.read_holding_registers(REG_INIT_STATE, count=1, slave=self.slave)
            if not r.isError() and r.registers[0] == 1:
                return
            time.sleep(0.2)
        raise TimeoutError("夹爪初始化超时")

    def set_force(self, percent: int) -> None:
        self.client.write_register(REG_FORCE, int(percent), slave=self.slave)

    def move(self, position_permille: int, wait: bool = True, timeout: float = 3.0) -> int:
        """position_permille: 0(全闭)~1000(全开)。返回最终夹持状态码。"""
        self.client.write_register(REG_POSITION, int(position_permille), slave=self.slave)
        if not wait:
            return 0
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            r = self.client.read_holding_registers(REG_GRIP_STATE, count=1, slave=self.slave)
            if not r.isError() and r.registers[0] != 0:
                return r.registers[0]
            time.sleep(0.05)
        raise TimeoutError("夹爪运动超时")

    def grasp(self, force_percent: int = 40) -> bool:
        """闭合夹取。返回是否夹住物体(状态码==2)。"""
        self.set_force(force_percent)
        return self.move(0) == 2

    def release(self) -> None:
        self.move(1000)

    def disconnect(self) -> None:
        self.client.close()
