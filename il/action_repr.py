"""观测/动作向量的编解码（方案二 D5/D6 决策）。

姿态用 rot6d（旋转矩阵前两列展平）表示，避免轴角/四元数的不连续性。
  observation.state (16,) = tcp_pos(3) + tcp_rot6d(6) + wrench(6) + gripper(1)
  action            (16,) = ref_pos(3) + ref_rot6d(6) + k(3) + kr(3) + gripper(1)
其中 wrench 为世界系、重力补偿并折算到 TCP 后的外部接触力；
ref = 遥操作者/策略给出的参考位姿（导纳环叠加柔顺偏移前的指令），
gripper ∈ [0,1] 为夹爪指令开度（千分比/1000）。
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

STATE_DIM = 16
ACTION_DIM = 16

STATE_NAMES = (
    [f"tcp_pos_{a}" for a in "xyz"] + [f"tcp_rot6d_{i}" for i in range(6)]
    + [f"wrench_{a}" for a in ("fx", "fy", "fz", "tx", "ty", "tz")] + ["gripper"])
ACTION_NAMES = (
    [f"ref_pos_{a}" for a in "xyz"] + [f"ref_rot6d_{i}" for i in range(6)]
    + [f"k_{a}" for a in "xyz"] + [f"kr_{a}" for a in "xyz"] + ["gripper"])


# ------------------------------------------------------------ rot6d
def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """旋转矩阵前两列按列展平 -> (6,)。"""
    return np.asarray(R, float)[:, :2].T.ravel()   # [r11,r21,r31, r12,r22,r32]


def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """Gram-Schmidt 正交化恢复旋转矩阵（对网络输出的非正交 6d 稳健）。"""
    a, b = np.asarray(r6, float)[:3], np.asarray(r6, float)[3:6]
    x = a / max(np.linalg.norm(a), 1e-9)
    b = b - (x @ b) * x
    y = b / max(np.linalg.norm(b), 1e-9)
    return np.column_stack([x, y, np.cross(x, y)])


def pose6_to_posrot6d(pose: np.ndarray) -> np.ndarray:
    """UR 位姿 [x,y,z,rx,ry,rz](轴角) -> (9,) pos+rot6d。"""
    pose = np.asarray(pose, float)
    return np.concatenate([pose[:3], matrix_to_rot6d(Rot.from_rotvec(pose[3:6]).as_matrix())])


def posrot6d_to_pose6(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    return np.concatenate([v[:3], Rot.from_matrix(rot6d_to_matrix(v[3:9])).as_rotvec()])


# ------------------------------------------------------------ 打包/解包
def pack_state(tcp_pose6: np.ndarray, wrench: np.ndarray, gripper: float) -> np.ndarray:
    return np.concatenate([pose6_to_posrot6d(tcp_pose6),
                           np.asarray(wrench, float), [float(gripper)]]).astype(np.float32)


def pack_action(ref_pose6: np.ndarray, k: np.ndarray, kr: np.ndarray,
                gripper: float) -> np.ndarray:
    return np.concatenate([pose6_to_posrot6d(ref_pose6), np.asarray(k, float),
                           np.asarray(kr, float), [float(gripper)]]).astype(np.float32)


def unpack_action(vec: np.ndarray) -> dict:
    """-> {pose6, k(3), kr(3), gripper}。"""
    v = np.asarray(vec, float)
    assert v.shape[-1] == ACTION_DIM, v.shape
    return {"pose6": posrot6d_to_pose6(v[:9]), "k": v[9:12].copy(),
            "kr": v[12:15].copy(), "gripper": float(v[15])}
