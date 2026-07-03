"""坐标系工具与任务系 {H} 管理（§1.3 / §4.1）。

位姿统一为 UR 约定 [x,y,z,rx,ry,rz]（轴角向量）。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as Rot


# ------------------------------------------------------------ 基础变换
def pose_to_Rp(pose) -> Tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(pose, float)
    return Rot.from_rotvec(pose[3:6]).as_matrix(), pose[:3].copy()


def Rp_to_pose(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.concatenate([p, Rot.from_matrix(R).as_rotvec()])


def compose(a, b) -> np.ndarray:
    """位姿复合 T_a ∘ T_b（pose6）。用于坐标链: 世界←UR10基座←TCP←孔。"""
    Ra, pa = pose_to_Rp(a)
    Rb, pb = pose_to_Rp(b)
    return Rp_to_pose(Ra @ Rb, pa + Ra @ pb)


def rotate_wrench(R: np.ndarray, w: np.ndarray) -> np.ndarray:
    """wrench 的纯旋转变换（不搬移参考点）：w 在 A 系，R = R_B_A，返回 B 系下的 w。"""
    return np.concatenate([R @ w[:3], R @ w[3:6]])


def shift_wrench(w: np.ndarray, r: np.ndarray) -> np.ndarray:
    """把 wrench 的力矩参考点沿 r 搬移（同一坐标系内）：T' = T − r × F。
    用途：把传感器原点的力矩折算到 TCP（柔顺中心设在工件尖端, §4.2）。"""
    return np.concatenate([w[:3], w[3:6] - np.cross(np.asarray(r, float), w[:3])])


# ------------------------------------------------------------ 任务系 {H}
class TaskFrame:
    """孔坐标系 {H}：原点在孔口中心，+Z 指向孔内（插入方向）。

    来源（§6 粗到精）：UR10 运动学+治具 CAD 计算，或 AprilTag 检测后刷新。
    """

    def __init__(self, pose_in_world: np.ndarray):
        self.R, self.p = pose_to_Rp(pose_in_world)

    def wrench_world_to_H(self, w_world: np.ndarray) -> np.ndarray:
        return rotate_wrench(self.R.T, w_world)

    def dir_to_world(self, v_h: np.ndarray) -> np.ndarray:
        return self.R @ np.asarray(v_h, float)

    def compose_target(self, ref_pose_world: np.ndarray,
                       dpos_h: np.ndarray, drot_h: np.ndarray) -> np.ndarray:
        """参考位姿 ⊕ {H} 系下的柔顺偏移 → 世界系 servoL 目标（§4.1 主环最后一步）。"""
        R_ref, p_ref = pose_to_Rp(ref_pose_world)
        p_t = p_ref + self.R @ np.asarray(dpos_h, float)
        dR_world = Rot.from_rotvec(self.R @ np.asarray(drot_h, float)).as_matrix()
        return Rp_to_pose(dR_world @ R_ref, p_t)

    def entry_pose(self, standoff: float, R_tool_in_H: np.ndarray | None = None) -> np.ndarray:
        """孔口上方 standoff 处的接近位姿：TCP 的 +Z(工具轴) 对齐 {H} 的 +Z。
        R_tool_in_H 可指定工具绕孔轴的朝向，默认绕 X 翻转（工具 Z 朝孔内）。"""
        if R_tool_in_H is None:
            R_tool_in_H = Rot.from_euler("x", 180, degrees=True).as_matrix()  # TODO: 按夹持姿态定
        R_tool = self.R @ R_tool_in_H
        p_tool = self.p - self.R[:, 2] * standoff
        return Rp_to_pose(R_tool, p_tool)
