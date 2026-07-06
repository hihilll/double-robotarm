"""双相机采集（腕部 D405 + 全局 D435），复用 drivers/camera.py（只读 import）。

frames() 返回 {name: rgb uint8 (out_h, out_w, 3)}；serial 为空的相机自动跳过并
警告（桌面无全局相机时可先单目采集, 但正式数据集必须双目——见采集 SOP #43）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

from drivers.camera import RealSenseCamera

CAM_NAMES = ("wrist", "global")


class CameraRig:
    def __init__(self, il_cfg: dict):
        cc = il_cfg["cameras"]
        self.out_wh = (cc["out_width"], cc["out_height"])
        self.cams: dict[str, RealSenseCamera] = {}
        for name in CAM_NAMES:
            c = cc[name]
            if not c["serial"]:
                print(f"[cameras] {name} 相机 serial 未配置, 跳过 ⚠ 正式采集必须配齐")
                continue
            self.cams[name] = RealSenseCamera(serial=c["serial"], width=c["width"],
                                              height=c["height"], fps=c["fps"],
                                              enable_depth=False)

    @property
    def names(self) -> list[str]:
        return list(self.cams)

    def frames(self) -> dict[str, np.ndarray]:
        assert cv2 is not None, "未安装 opencv-python"
        out = {}
        for name, cam in self.cams.items():
            rgb, _ = cam.frame()
            out[name] = cv2.resize(rgb, self.out_wh, interpolation=cv2.INTER_AREA)
        return out

    def close(self) -> None:
        for cam in self.cams.values():
            cam.close()
