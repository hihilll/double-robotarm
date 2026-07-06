"""遥操作主环（任务#42）：SpaceMouse → 参考位姿 → 导纳叠加 → servoL @125Hz。

设计（方案二 §2/§3）:
  - 操作者用 SpaceMouse 速度积分驱动"参考位姿 ref"；导纳环在 ref 上叠加柔顺偏移
    （接触力被吸收, 演示数据天然带柔顺行为）；
  - 按键0 循环切换刚度档 low→mid→high（Comp-ACT 档位机制），set_params 热切不跳变；
  - 按键1 夹爪开/合切换；
  - 世界系即任务系（TaskFrame(0)）——采集与部署保持同一 wrench 参考系；
  - 每周期把快照写入 LatestValue，供 25Hz 录制线程无锁读取（§1.4 总线约定）。

⚠ 首次真机运行: 示教器速度滑块 30%、手扶急停、末端周围留空; 逐轴核对
  axis_sign 后再靠近工件。安全层(方案一 SafetyMonitor)每周期检查, 违规即停。

单独手感调试:  python il/teleop/teleop_loop.py --payload none
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from control.admittance import AdmittanceController
from control.safety import SafetyError
from control.task_frame import TaskFrame
from drivers.bus import LatestValue
from il.adapters import (MODE_CYCLE, load_configs, load_il_config,
                         params_from_stiffness, stiffness_of_mode)
from il.teleop.spacemouse import SpaceMouse


class TeleopSession:
    """一次遥操作会话。step() 由调用方以 125Hz 驱动（录制脚本放在独立线程）。"""

    def __init__(self, stack, il_cfg: dict, gripper=None):
        self.stack = stack
        self.arm = stack.arm
        self.prim = stack.prim
        self.cfg = il_cfg
        self.tc = il_cfg["teleop"]
        self.gripper = gripper
        _, ac = load_configs()
        self.limits = ac["limits"]
        self.ws_min = np.asarray(ac["safety"]["workspace_min"], float) + 0.01
        self.ws_max = np.asarray(ac["safety"]["workspace_max"], float) - 0.01

        self.sm = SpaceMouse(self.tc["deadzone"], self.tc["axis_sign"])
        self.task = TaskFrame(np.zeros(6))          # 任务系 = 世界系
        self.mode_idx = 0
        k, kr = stiffness_of_mode(il_cfg, self.mode)
        self.k, self.kr = k, kr
        self.adm = AdmittanceController(params_from_stiffness(k, kr, il_cfg, self.limits))
        self.grip_cmd = 1.0                          # 1=全开, 0=全闭(归一化千分比)
        self.snapshot = LatestValue()                # 录制线程读这里
        self.ref: np.ndarray | None = None
        self.running = False
        self.error: str | None = None

    # ------------------------------------------------------------
    @property
    def mode(self) -> str:
        return MODE_CYCLE[self.mode_idx]

    def reset(self) -> None:
        """以当前 TCP 为参考位姿开始新会话。"""
        self.ref = self.arm.tcp_pose()
        self.adm.reset()
        self.mode_idx = 0
        self._set_mode(0)
        self.running = True
        self.error = None

    def _set_mode(self, idx: int) -> None:
        self.mode_idx = idx % len(MODE_CYCLE)
        self.k, self.kr = stiffness_of_mode(self.cfg, self.mode)
        self.adm.set_params(params_from_stiffness(self.k, self.kr, self.cfg, self.limits),
                            keep_state=True)

    def _toggle_gripper(self) -> None:
        close = self.tc["gripper_close_permille"] / 1000.0
        self.grip_cmd = close if self.grip_cmd > 0.5 else 1.0
        if self.gripper is not None:
            self.gripper.move(int(self.grip_cmd * 1000), wait=False)  # 不阻塞主环

    # ------------------------------------------------------------ 125Hz 周期
    def step(self) -> None:
        t0 = self.arm.init_period()
        dt = self.arm.dt

        vel, pressed = self.sm.read()
        if self.tc["buttons"]["mode_cycle"] in pressed:
            self._set_mode(self.mode_idx + 1)
        if self.tc["buttons"]["gripper_toggle"] in pressed:
            self._toggle_gripper()

        # 参考位姿积分（世界系平动 + 世界系转动）
        self.ref[:3] += vel[:3] * self.tc["v_lin_max"] * dt
        self.ref[:3] = np.clip(self.ref[:3], self.ws_min, self.ws_max)
        drot = vel[3:6] * self.tc["v_rot_max"] * dt
        if np.any(drot):
            from scipy.spatial.transform import Rotation as Rot
            R = Rot.from_rotvec(drot).as_matrix() @ Rot.from_rotvec(self.ref[3:6]).as_matrix()
            self.ref[3:6] = Rot.from_matrix(R).as_rotvec()

        tcp = self.arm.tcp_pose()
        # 拴绳: ref 离实际 TCP 过远(被挡/卡住)则拉回, 防止松开后飞扑
        leash = self.tc["leash_mm"] * 1e-3
        gap = self.ref[:3] - tcp[:3]
        d = np.linalg.norm(gap)
        if d > leash:
            self.ref[:3] = tcp[:3] + gap / d * leash

        w, age = self.prim.wrench_H(self.task)       # 世界系接触 wrench(已补偿/折算TCP)
        if self.tc["admittance_on"]:
            dpos, dr = self.adm.step(w, dt)
            target = self.task.compose_target(self.ref, dpos, dr)
        else:
            target = self.ref.copy()

        self.prim.safety.check(w, age, tcp, target)
        self.arm.servo_l(target)

        self.snapshot.put({"t": time.monotonic(), "tcp": tcp, "ref": self.ref.copy(),
                           "target": target, "wrench": w, "k": self.k.copy(),
                           "kr": self.kr.copy(), "grip": self.grip_cmd,
                           "mode": self.mode})
        self.arm.wait_period(t0)

    def run(self, stop_flag=None) -> None:
        """阻塞跑主环直到 stop_flag() 为真或安全违规。供录制脚本线程调用。"""
        try:
            while self.running and (stop_flag is None or not stop_flag()):
                self.step()
        except SafetyError as e:
            self.error = str(e)
            print(f"\n[teleop] 安全停止: {e}")
        finally:
            self.running = False
            self.arm.servo_stop()

    def close(self) -> None:
        self.running = False
        self.sm.close()


def main() -> None:
    from il.adapters import build_gripper, build_ur5_stack

    ap = argparse.ArgumentParser(description="遥操作手感调试(不录数据)")
    ap.add_argument("--payload", default="none")
    ap.add_argument("--gripper", action="store_true", help="接入 UR5 夹爪")
    args = ap.parse_args()

    il_cfg = load_il_config()
    stack = build_ur5_stack(payload=args.payload)
    grip = build_gripper(stack.rc, "ur5") if args.gripper else None
    session = TeleopSession(stack, il_cfg, gripper=grip)
    print("⚠ 速度滑块 30%、手扶急停。SpaceMouse 键0=切刚度档 键1=夹爪, Ctrl+C 退出。")
    input("确认末端周围无障碍, 回车开始...")
    session.reset()
    try:
        session.run()
    except KeyboardInterrupt:
        print("\n退出")
    finally:
        session.close()
        stack.close()


if __name__ == "__main__":
    main()
