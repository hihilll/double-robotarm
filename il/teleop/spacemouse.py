"""SpaceMouse Compact 输入封装（任务#41）。

依赖 pyspacemouse (pip install pyspacemouse; Linux 需 hidapi + udev 规则,
见 il/README.md)。read() 返回 (vel6 ∈[-1,1] 已去死区加符号, 按键沿触发集合)。
⚠ 真机首测: 逐轴推动核对 config_il.yaml teleop.axis_sign 的符号方向。
"""
from __future__ import annotations

import numpy as np

try:
    import pyspacemouse
except ImportError:
    pyspacemouse = None  # type: ignore


class SpaceMouse:
    def __init__(self, deadzone: float = 0.12, axis_sign=(1, 1, 1, 1, 1, 1)):
        if pyspacemouse is None:
            raise RuntimeError("未安装 pyspacemouse: pip install pyspacemouse "
                               "(Linux 另需 sudo apt install libhidapi-dev + udev 规则)")
        self.deadzone = deadzone
        self.sign = np.asarray(axis_sign, float)
        self._prev_buttons: list[int] = []
        assert pyspacemouse.open(), "SpaceMouse 打开失败(接口/权限?)"

    def read(self) -> tuple[np.ndarray, set[int]]:
        """非阻塞读最新状态。返回 (vel6, 本次新按下的按键序号集合)。
        vel6 顺序 [x,y,z,rx,ry,rz]，已去死区、重标定到 [-1,1]、乘符号。"""
        st = pyspacemouse.read()
        raw = np.array([st.x, st.y, st.z, st.roll, st.pitch, st.yaw], float)
        out = np.zeros(6)
        big = np.abs(raw) > self.deadzone
        out[big] = (np.abs(raw[big]) - self.deadzone) / (1.0 - self.deadzone) * np.sign(raw[big])
        buttons = list(getattr(st, "buttons", []) or [])
        pressed = {i for i, b in enumerate(buttons)
                   if b and (i >= len(self._prev_buttons) or not self._prev_buttons[i])}
        self._prev_buttons = buttons
        return out * self.sign, pressed

    def close(self) -> None:
        pyspacemouse.close()
