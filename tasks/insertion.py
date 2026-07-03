"""任务#28: 通用插入状态机（plug_insert 与 screw_place 共用, §5.1）。

流程: APPROACH → CONTACT → SEARCH → ALIGN_INSERT → VERIFY
失败 → RECOVER(退回孔口上方) → RETRY(≤retries 次)。
所有阈值来自 admittance.yaml 的 tasks.<name> 与 state_machine, 真机只调 yaml。
"""
from __future__ import annotations

import time

import numpy as np

from common.setup import UR5Stack, admittance_params
from control.task_frame import TaskFrame


class InsertionTask:
    def __init__(self, stack: UR5Stack, task_name: str, gripper=None):
        self.stack = stack
        self.prim = stack.prim
        self.gripper = gripper               # drivers.gripper.DHGripper | None
        self.cfg = stack.ac["tasks"][task_name]
        self.sm = stack.ac["state_machine"]
        self.search_params = admittance_params(stack.ac, "search")
        self.insert_params = admittance_params(stack.ac, "insert")
        self.task_name = task_name

    def run(self, H: TaskFrame) -> dict:
        """执行一次插入, 返回评测用 info 字典(#34 依赖此格式)。"""
        cfg, sm = self.cfg, self.sm
        info = {"task": self.task_name, "success": False, "attempts": 0,
                "fail_stage": None, "t_total": 0.0, "t_search": 0.0}
        entry = H.entry_pose(sm["standoff"])
        t0 = time.monotonic()

        for attempt in range(1, sm["retries"] + 1):
            info["attempts"] = attempt
            # ---- APPROACH ----
            self.prim.move_free(entry, v=sm["approach_v"])
            # ---- CONTACT ----
            try:
                self.prim.guarded_move(H, np.array([0, 0, 1.0]),
                                       sm["contact_v"], cfg["contact_force"])
            except TimeoutError:
                info["fail_stage"] = "contact"
                continue                        # RECOVER: 下轮重新 APPROACH
            # ---- SEARCH ----
            ts = time.monotonic()
            dropped = self.prim.spiral_search(
                H, self.search_params, f_push=cfg["search_force"],
                pitch=cfg["spiral_pitch"], v=cfg["spiral_v"],
                r_max=cfg["spiral_r_max"], drop_dz=cfg["drop_dz"])
            info["t_search"] += time.monotonic() - ts
            if not dropped:
                info["fail_stage"] = "search"
                self._recover(entry)
                continue
            # ---- ALIGN_INSERT ----
            inserted = self.prim.compliant_insert(
                H, self.insert_params, f_insert=cfg["insert_force"],
                depth_goal=cfg["insert_depth"])
            if not inserted:
                info["fail_stage"] = "insert"
                self._recover(entry)
                continue
            # ---- VERIFY + 释放退出 ----
            # 简版: 深度达标即成功。插头卡合回拉验证(2N)真机调好后在此补充 TODO
            if self.gripper is not None:
                self.gripper.release()       # 工件留在孔内, 松爪
            self._recover(entry)             # 退回孔口上方
            info["success"], info["fail_stage"] = True, None
            break

        info["t_total"] = time.monotonic() - t0
        return info

    def _recover(self, entry: np.ndarray) -> None:
        """RECOVER: 缓慢退回孔口上方接近位姿。"""
        self.prim.move_free(entry, v=0.02)
