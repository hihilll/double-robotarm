"""任务#32: 插头拔出（§5.2）。

流程: 移到孔口上方 → 沿孔轴前进到夹持位(grasp_advance) → 闭合夹爪 →
compliant_extract 力斜坡拉拔 → 退出。阈值全部来自 admittance.yaml tasks.plug_extract。
"""
from __future__ import annotations

import time

import numpy as np

from common.setup import UR5Stack, admittance_params
from control.task_frame import TaskFrame


class PlugExtractTask:
    def __init__(self, stack: UR5Stack, gripper=None):
        self.stack = stack
        self.prim = stack.prim
        self.gripper = gripper                 # drivers.gripper.DHGripper | None(手动夹持调试)
        self.cfg = stack.ac["tasks"]["plug_extract"]
        self.sm = stack.ac["state_machine"]
        self.params = admittance_params(stack.ac, "insert")   # 拔出复用 insert 柔顺参数

    def run(self, H: TaskFrame) -> dict:
        cfg, sm = self.cfg, self.sm
        info = {"task": "plug_extract", "success": False, "attempts": 1,
                "fail_stage": None, "t_total": 0.0, "t_search": 0.0}
        t0 = time.monotonic()

        # ---- 接近并到达夹持位 ----
        entry = H.entry_pose(sm["standoff"])
        self.prim.move_free(entry, v=sm["approach_v"])
        grasp = H.compose_target(entry, np.array([0, 0, sm["standoff"] + cfg["grasp_advance"]]),
                                 np.zeros(3))
        self.prim.move_free(grasp, v=0.02)
        # ---- 夹持 ----
        if self.gripper is not None:
            if not self.gripper.grasp(force_percent=60):
                info["fail_stage"] = "grasp"
                info["t_total"] = time.monotonic() - t0
                return info
        else:
            input("  [无夹爪调试] 手动确认已夹持, 回车继续...")
        # ---- 力斜坡拔出 ----
        ok = self.prim.compliant_extract(
            H, self.params, f_start=cfg["f_start"], f_rate=cfg["f_rate"],
            f_max=cfg["f_max"], extract_dist=cfg["extract_dist"],
            wiggle_above=cfg["wiggle_above"])
        info["success"] = ok
        info["fail_stage"] = None if ok else "extract"
        # ---- 退出 ----
        self.prim.move_free(entry, v=0.05)
        info["t_total"] = time.monotonic() - t0
        return info
