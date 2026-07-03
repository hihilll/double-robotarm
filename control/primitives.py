"""操作原语（§4.3）：所有任务状态机用这 5 个原语拼装。

每个柔顺原语内部自带 125Hz 主环（§4.1 伪码的实现），模式:
    t0 = arm.init_period()
    wrench_H = 力管线(读FT → 重力补偿 → 折算TCP → 旋到{H})
    dpos, drot = admittance.step(wrench_H, dt, wrench_ref)
    target = task.compose_target(ref, dpos, drot)
    safety.check(...);  arm.servo_l(target);  arm.wait_period(t0)

⚠ 符号约定（真机首次运行必须逐项校核, 低速+30%速度滑块）:
  - {H} +Z 指向孔内 ⇒ 推进方向为 +Z_H, 期望接触反力为 −Z_H
  - 传感器输出是"传感器受到的力"还是"施加给环境的力"因型号而异, 决定 wrench 正负号
"""
from __future__ import annotations

import math
import time

import numpy as np

from control.admittance import AdmittanceController, AdmittanceParams
from control.safety import SafetyError, SafetyMonitor
from control.task_frame import TaskFrame, pose_to_Rp, rotate_wrench, shift_wrench
from drivers.ft_sensor import FTReader, GravityCompensator
from drivers.ur_arm import URArm


class Primitives:
    def __init__(self, arm: URArm, ft: FTReader, gravity: GravityCompensator,
                 safety: SafetyMonitor, R_tcp_sensor: np.ndarray,
                 p_sensor_to_tcp: np.ndarray):
        self.arm = arm
        self.ft = ft
        self.gravity = gravity
        self.safety = safety
        self.R_tcp_sensor = np.asarray(R_tcp_sensor, float)
        self.p_sensor_to_tcp = np.asarray(p_sensor_to_tcp, float)
        self.logger = None  # 任务层注入 common.logger.RunLogger, 各原语主环自动记录

    def _log(self, state: str, w: np.ndarray, target: np.ndarray) -> None:
        if self.logger is not None:
            self.logger.log(state=state, wrench_h=w, target=target,
                            tcp=self.arm.tcp_pose())

    # ------------------------------------------------------------ 力管线
    def wrench_H(self, task: TaskFrame) -> tuple[np.ndarray, float]:
        """传感器原始值 → {H} 系下折算到 TCP 的外部接触 wrench。返回 (wrench, 数据年龄)。"""
        raw, age = self.ft.latest()
        if raw is None:
            return np.zeros(6), age
        R_base_tcp, _ = pose_to_Rp(self.arm.tcp_pose())
        w_s = self.gravity.compensate(raw, R_base_tcp)          # 传感器系, 已去重力/零偏
        w_s = shift_wrench(w_s, self.p_sensor_to_tcp)           # 力矩折算到 TCP(柔顺中心)
        w_base = rotate_wrench(R_base_tcp @ self.R_tcp_sensor, w_s)
        return task.wrench_world_to_H(w_base), age

    # ------------------------------------------------------------ 原语 1
    def move_free(self, pose: np.ndarray, v: float = 0.1, a: float = 0.5) -> None:
        """纯位置运动，高刚度。APPROACH 阶段用。"""
        self.arm.move_l(pose, v, a, blocking=True)

    # ------------------------------------------------------------ 原语 2
    def guarded_move(self, task: TaskFrame, direction_h: np.ndarray,
                     speed: float, f_stop: float, timeout: float = 10.0) -> np.ndarray:
        """沿 {H} 系方向低速前进，运动方向上的接触反力超过 f_stop 即停（CONTACT 阶段）。
        返回停止时的 TCP 位姿；超时抛 TimeoutError。纯位置推进，不叠加导纳。"""
        d = np.asarray(direction_h, float)
        d = d / np.linalg.norm(d)
        ref = self.arm.tcp_pose()
        traveled, t_start = 0.0, time.monotonic()
        try:
            while time.monotonic() - t_start < timeout:
                t0 = self.arm.init_period()
                w, age = self.wrench_H(task)
                if -(w[:3] @ d) > f_stop:               # 反力沿 -d 方向
                    self.arm.servo_stop()
                    return self.arm.tcp_pose()
                traveled += speed * self.arm.dt
                target = task.compose_target(ref, d * traveled, np.zeros(3))
                self.safety.check(w, age, self.arm.tcp_pose(), target)
                self.arm.servo_l(target)
                self._log("guarded_move", w, target)
                self.arm.wait_period(t0)
        except SafetyError:
            self._abort(task)
            raise
        self.arm.servo_stop()
        raise TimeoutError("guarded_move 超时未接触")

    # ------------------------------------------------------------ 原语 3+4
    def spiral_search(self, task: TaskFrame, params: AdmittanceParams,
                      f_push: float, pitch: float, v: float, r_max: float,
                      drop_dz: float, timeout: float = 20.0) -> bool:
        """Z_H 恒力按压(hold_force) + XY 面阿基米德螺旋(§5.1 SEARCH)。
        落入判据: Z 向位置突进 > drop_dz 且 Z 向反力骤降 50%。返回是否落入。"""
        adm = AdmittanceController(params)
        wrench_ref = np.array([0, 0, -f_push, 0, 0, 0])   # 期望 -Z_H 反力 = 推进 f_push
        ref = self.arm.tcp_pose()
        theta, t_start = 0.0, time.monotonic()
        z_window: list[float] = []                        # 最近 ~100ms 的 Z_H 深度
        try:
            while time.monotonic() - t_start < timeout:
                t0 = self.arm.init_period()
                w, age = self.wrench_H(task)
                dpos, drot = adm.step(w, self.arm.dt, wrench_ref)

                # 螺旋轨迹叠加在导纳偏移之上（XY 通道）
                r = pitch * theta / (2 * math.pi)
                if r < r_max:
                    omega = v / max(r, 1e-4)              # 恒线速度 ⇒ ω = v/r
                    theta += omega * self.arm.dt
                spiral = np.array([r * math.cos(theta), r * math.sin(theta), 0.0])

                target = task.compose_target(ref, dpos + spiral, drot)
                self.safety.check(w, age, self.arm.tcp_pose(), target)
                self.arm.servo_l(target)
                self._log("search", w, target)

                # ---- 落入检测：Z_H 深度窗口 + 力骤降 ----
                depth = self._depth_in_H(task, ref)
                z_window.append(depth)
                if len(z_window) > 13:                    # ≈100ms @125Hz
                    z_window.pop(0)
                    dz = z_window[-1] - z_window[0]
                    fz_drop = -w[2] < 0.5 * f_push        # 反力剩不到一半
                    if dz > drop_dz and fz_drop:
                        self.arm.servo_stop()
                        return True
                self.arm.wait_period(t0)
        except SafetyError:
            self._abort(task)
            raise
        self.arm.servo_stop()
        return False

    # ------------------------------------------------------------ 原语 5
    def compliant_insert(self, task: TaskFrame, params: AdmittanceParams,
                         f_insert: float, depth_goal: float, timeout: float = 15.0,
                         stall_time: float = 0.5, stall_lat_f: float = 8.0,
                         wiggle_amp_deg: float = 1.0, wiggle_freq: float = 2.0) -> bool:
        """落入后柔顺下插(§5.1 ALIGN_INSERT)：Z_H 恒力, 转动XY低刚度(软件RCC)。
        卡阻处理(#27): stall_time 内深度无进展且横向力>stall_lat_f 时,
        沿横向力矩方向叠加 ±wiggle_amp_deg°@wiggle_freq Hz 正弦摆动。"""
        adm = AdmittanceController(params)
        wrench_ref = np.array([0, 0, -f_insert, 0, 0, 0])
        ref = self.arm.tcp_pose()
        t_start = time.monotonic()
        best_depth, t_progress = 0.0, t_start
        try:
            while time.monotonic() - t_start < timeout:
                t0 = self.arm.init_period()
                now = time.monotonic()
                w, age = self.wrench_H(task)
                dpos, drot = adm.step(w, self.arm.dt, wrench_ref)

                depth = self._depth_in_H(task, ref)
                if depth > best_depth + 1e-4:
                    best_depth, t_progress = depth, now
                # ---- 卡阻检测 → 摆动 ----
                wiggle = np.zeros(3)
                lat_f = float(np.linalg.norm(w[:2]))
                if now - t_progress > stall_time and lat_f > stall_lat_f:
                    tau = w[3:5]
                    axis = tau / max(np.linalg.norm(tau), 1e-6)
                    amp = math.radians(wiggle_amp_deg)
                    s = amp * math.sin(2 * math.pi * wiggle_freq * (now - t_start))
                    wiggle = np.array([axis[0] * s, axis[1] * s, 0.0])

                target = task.compose_target(ref, dpos, drot + wiggle)
                self.safety.check(w, age, self.arm.tcp_pose(), target)
                self.arm.servo_l(target)
                self._log("insert", w, target)
                if depth >= depth_goal:
                    self.arm.servo_stop()
                    return True
                self.arm.wait_period(t0)
        except SafetyError:
            self._abort(task)
            raise
        self.arm.servo_stop()
        return False

    # ------------------------------------------------------------ 原语 6
    def compliant_extract(self, task: TaskFrame, params: AdmittanceParams,
                          f_start: float, f_rate: float, f_max: float,
                          extract_dist: float, wiggle_above: float = 25.0,
                          timeout: float = 20.0) -> bool:
        """柔顺拔出(§5.2, #32): 沿 -Z_H 力斜坡拉拔(f_start 起每秒 +f_rate, 上限 f_max),
        XY 与转动低刚度跟随; 拉力 ≥wiggle_above 仍不动时叠加 ±1.5°@2Hz 摆动。
        成功判据: 沿 -Z_H 位移 > extract_dist 且拉拔反力骤降。
        ⚠ 拔出方向 wrench_ref 符号与插入相反, 真机首次运行低速校核。"""
        adm = AdmittanceController(params)
        ref = self.arm.tcp_pose()
        t_start = time.monotonic()
        try:
            while time.monotonic() - t_start < timeout:
                t0 = self.arm.init_period()
                now = time.monotonic()
                f_pull = min(f_start + f_rate * (now - t_start), f_max)
                wrench_ref = np.array([0, 0, +f_pull, 0, 0, 0])
                w, age = self.wrench_H(task)
                dpos, drot = adm.step(w, self.arm.dt, wrench_ref)

                wiggle = np.zeros(3)
                if f_pull >= wiggle_above:
                    amp = math.radians(1.5)
                    wiggle[0] = amp * math.sin(2 * math.pi * 2.0 * (now - t_start))

                target = task.compose_target(ref, dpos, drot + wiggle)
                self.safety.check(w, age, self.arm.tcp_pose(), target)
                self.arm.servo_l(target)
                self._log("extract", w, target)
                pulled = -self._depth_in_H(task, ref)
                if pulled > extract_dist and abs(w[2]) < 0.3 * f_pull:
                    self.arm.servo_stop()
                    return True
                self.arm.wait_period(t0)
        except SafetyError:
            self._abort(task)
            raise
        self.arm.servo_stop()
        return False

    # ------------------------------------------------------------ 内部
    def _depth_in_H(self, task: TaskFrame, ref_pose: np.ndarray) -> float:
        """当前 TCP 相对参考位姿沿 {H}+Z 的推进深度。"""
        dp = self.arm.tcp_pose()[:3] - ref_pose[:3]
        return float(dp @ task.R[:, 2])

    def _abort(self, task: TaskFrame) -> None:
        """安全违规处理: 停伺服并沿 -Z_H 抬升 5mm 脱离接触。"""
        self.arm.servo_stop()
        back = self.arm.tcp_pose()
        back[:3] -= task.R[:, 2] * 0.005
        self.arm.move_l(back, v=0.02)
