"""任务#31: 双臂主状态机（§7.2）+ 任务#30: 视觉粗到精接入。

流程:  UR10 [pick→]present → 计算{H}(UR10运动学∘治具CAD) → [视觉刷新{H}+一致性检查]
       → UR5 执行任务 → [UR10 place] → 返回评测 info

pick/place 依赖现场示教(robots.yaml ur10.pick_pose), 为 null 时跳过(工件预先夹好)。

用法(单次):   python tasks/dual_arm_manager.py --task plug_insert
     (带视觉): python tasks/dual_arm_manager.py --task plug_insert --vision
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.logger import RunLogger
from common.setup import build_gripper, build_ur5_stack, build_ur10
from control.task_frame import TaskFrame, compose
from tasks.insertion import InsertionTask
from tasks.plug_extract import PlugExtractTask

PAYLOAD_OF = {"plug_insert": "plug", "plug_extract": "none", "screw_place": "screw"}


class DualArmManager:
    def __init__(self, task_name: str, use_vision: bool = False, log: bool = True,
                 use_grippers: bool = True):
        self.task_name = task_name
        self.use_vision = use_vision
        self.stack = build_ur5_stack(payload=PAYLOAD_OF[task_name])
        self.ur10, self.rc = build_ur10()
        self.ur5_gripper = build_gripper(self.rc, "ur5") if use_grippers else None
        self.ur10_gripper = build_gripper(self.rc, "ur10") if use_grippers else None
        self.sm = self.stack.ac["state_machine"]
        if log:
            self.stack.prim.logger = RunLogger(task_name)
        self._locator = None
        if use_vision:
            from drivers.camera import RealSenseCamera
            from perception.tag_locator import TagLocator
            cc = self.rc["camera"]
            assert cc["T_tcp_cam"] is not None, "先完成手眼标定(#20)"
            cam = RealSenseCamera(**cc["wrist"], enable_depth=False)
            self._locator = TagLocator(cam, cc["tag"]["family"], cc["tag"]["size"],
                                       np.asarray(cc["T_tcp_cam"], float))

    # ------------------------------------------------------------ {H} 计算
    def kinematic_H(self) -> np.ndarray:
        """{H} = T_w_B10 ∘ T_B10_tcp10 ∘ T_tcp10_hole（首选来源, §6）。"""
        T_w_b10 = np.asarray(self.rc["dual_arm_calib"]["pose_b10_in_world"], float)
        T_tcp_hole = np.asarray(
            self.rc["task_geometry"][self.task_name]["hole_in_ur10_tcp"], float)
        return compose(compose(T_w_b10, self.ur10.tcp_pose()), T_tcp_hole)

    def refined_H(self, H_kin: np.ndarray) -> np.ndarray:
        """#30: 视觉刷新 {H} 并与运动学值做一致性检查。"""
        T_tag_hole = self.rc["camera"]["tag"]["T_tag_hole"][self.task_name]
        H_vis = self._locator.locate_H(self.stack.arm.tcp_pose(), T_tag_hole)
        dev = np.linalg.norm(H_vis[:3] - H_kin[:3])
        if dev > self.sm["vision_consistency"]:
            raise RuntimeError(f"视觉/运动学 {{H}} 偏差 {dev * 1000:.1f}mm 超限, "
                               f"检查标定或 tag 贴装后重试")
        return H_vis

    # ------------------------------------------------------------ 单次执行
    def run_episode(self, present_perturb: np.ndarray | None = None) -> dict:
        c10 = self.rc["ur10"]
        # UR10: [pick →] present（可叠加评测扰动 #34）
        if c10.get("pick_pose"):
            self.ur10.move_l(np.asarray(c10["pick_pose"], float), 0.1, 0.3)
            if self.ur10_gripper is not None and not self.ur10_gripper.grasp(force_percent=80):
                raise RuntimeError("UR10 抓取工件失败(夹爪状态≠夹住)")
        present = np.asarray(c10["present_pose"], float)
        if present_perturb is not None:
            present = present + present_perturb
        self.ur10.move_l(present, 0.1, 0.3)

        H_pose = self.kinematic_H()
        if self._locator is not None:
            hover = TaskFrame(H_pose).entry_pose(0.15)   # 悬停 15cm 精定位
            self.stack.prim.move_free(hover, v=self.sm["approach_v"])
            H_pose = self.refined_H(H_pose)
        H = TaskFrame(H_pose)

        # UR5: 执行任务
        if self.task_name == "plug_extract":
            info = PlugExtractTask(self.stack, gripper=self.ur5_gripper).run(H)
        else:
            info = InsertionTask(self.stack, self.task_name,
                                 gripper=self.ur5_gripper).run(H)

        # UR10: 放回料位(与 pick 同位姿)
        if c10.get("pick_pose"):
            self.ur10.move_l(np.asarray(c10["pick_pose"], float), 0.1, 0.3)
            if self.ur10_gripper is not None:
                self.ur10_gripper.release()
        return info

    def close(self) -> None:
        if self.stack.prim.logger is not None:
            self.stack.prim.logger.close()
        for g in (self.ur5_gripper, self.ur10_gripper):
            if g is not None:
                g.disconnect()
        self.stack.close()
        self.ur10.disconnect()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(PAYLOAD_OF), required=True)
    ap.add_argument("--vision", action="store_true")
    args = ap.parse_args()

    mgr = DualArmManager(args.task, use_vision=args.vision)
    try:
        info = mgr.run_episode()
        print(f"\n结果: {info}")
    finally:
        mgr.close()


if __name__ == "__main__":
    main()
