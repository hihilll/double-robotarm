"""双臂硬件聚合 + LeRobotDataset v3.0 的 schema/写入辅助（任务#39）。

不实现 lerobot 的 Robot 插件基类（其 API 随版本变动），而是：
  - DualArmRig 聚合 UR5栈/UR10/夹爪/相机（全部只读 import 方案一驱动）；
  - dataset_features() 给出与 il/action_repr.py 一致的特征 schema；
  - create_or_resume_dataset() / add_frame_compat() 封装 lerobot 数据集 API,
    并对 v0.4 前后 add_frame 签名差异做兼容。
lerobot 仅在真正建数据集时才 import——控制侧模块不依赖它。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from il import action_repr as ar
from il.adapters import ROOT, build_gripper, build_ur5_stack, build_ur10, load_il_config
from il.learning.cameras import CameraRig


# ------------------------------------------------------------ 硬件聚合
class DualArmRig:
    """采集/部署共用的硬件容器。use_ur10=False 时为桌面台架模式(治具固定)。"""

    def __init__(self, il_cfg: dict, payload: str = "none", use_ur10: bool = True,
                 use_cameras: bool = True, use_gripper: bool = True):
        self.cfg = il_cfg
        self.stack = build_ur5_stack(payload=payload)
        self.rc = self.stack.rc
        self.ur10 = None
        if use_ur10:
            self.ur10, _ = build_ur10()
        self.gripper = build_gripper(self.rc, "ur5") if use_gripper else None
        self.cameras = CameraRig(il_cfg) if use_cameras else None

    def present_ur10(self, task_name: str, rng: np.random.Generator) -> np.ndarray:
        """UR10 到呈现位姿并叠加采集随机扰动。返回实际下发的位姿(记入 episode 元数据)。"""
        assert self.ur10 is not None, "台架模式无 UR10"
        t = self.cfg["tasks"][task_name]
        present = np.asarray(self.rc["ur10"]["present_pose"], float)
        present[:3] += rng.uniform(-1, 1, 3) * t["perturb_mm"] * 1e-3
        present[3:6] += rng.uniform(-1, 1, 3) * np.radians(t["perturb_deg"])
        self.ur10.move_l(present, 0.1, 0.3)
        return present

    def close(self) -> None:
        if self.cameras is not None:
            self.cameras.close()
        if self.gripper is not None:
            self.gripper.disconnect()
        if self.ur10 is not None:
            self.ur10.disconnect()
        self.stack.close()


# ------------------------------------------------------------ 数据集 schema
OBS_STATE = "observation.state"
ACTION = "action"


def image_key(cam_name: str) -> str:
    return f"observation.images.{cam_name}"


def dataset_features(il_cfg: dict, cam_names: list[str]) -> dict:
    """LeRobotDataset.create(features=...) 的 schema, 与 action_repr 严格一致。"""
    h, w = il_cfg["cameras"]["out_height"], il_cfg["cameras"]["out_width"]
    feats = {
        OBS_STATE: {"dtype": "float32", "shape": (ar.STATE_DIM,),
                    "names": ar.STATE_NAMES},
        ACTION: {"dtype": "float32", "shape": (ar.ACTION_DIM,),
                 "names": ar.ACTION_NAMES},
    }
    for name in cam_names:
        feats[image_key(name)] = {"dtype": "video", "shape": (h, w, 3),
                                  "names": ["height", "width", "channels"]}
    return feats


# ------------------------------------------------------------ lerobot 封装
def create_or_resume_dataset(repo_id: str, il_cfg: dict, cam_names: list[str]):
    """本地建/续 LeRobotDataset v3.0（不联网, 不推 Hub）。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = ROOT / il_cfg["dataset"]["root"] / repo_id
    if root.exists():
        ds = LeRobotDataset(repo_id, root=root)
        print(f"[dataset] 续采 {root} (已有 {ds.num_episodes} 条)")
        return ds
    root.parent.mkdir(parents=True, exist_ok=True)
    return LeRobotDataset.create(repo_id, fps=int(il_cfg["dataset"]["fps"]), root=root,
                                 features=dataset_features(il_cfg, cam_names))


def add_frame_compat(ds, frame: dict, task: str) -> None:
    """兼容 lerobot 各小版本 add_frame 签名（task 在 frame 内 / 关键字参数）。"""
    try:
        ds.add_frame(frame, task=task)
    except TypeError:
        ds.add_frame({**frame, "task": task})
