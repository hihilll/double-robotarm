"""任务#29: AprilTag → 孔坐标系 {H}（§6-A）。

坐标链:  T_world_H = T_world_tcp ∘ T_tcp_cam ∘ T_cam_tag ∘ T_tag_hole
其中 T_tcp_cam 来自手眼标定(#20), T_tag_hole 由治具 CAD 离线量出(robots.yaml)。
多帧中位数滤波抑制单帧抖动。
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
    from pupil_apriltags import Detector
except ImportError:
    cv2 = Detector = None  # type: ignore

from control.task_frame import Rp_to_pose, compose


class TagLocator:
    def __init__(self, camera, family: str, tag_size: float,
                 T_tcp_cam: np.ndarray, tag_id: int | None = None):
        if Detector is None:
            raise RuntimeError("未安装依赖: pip install pupil-apriltags opencv-python")
        self.camera = camera
        self.tag_size = tag_size
        self.tag_id = tag_id
        self.T_tcp_cam = np.asarray(T_tcp_cam, float)
        i = camera.intrinsics()
        self._cam_params = (i["fx"], i["fy"], i["cx"], i["cy"])
        self._det = Detector(families=family)

    def detect_tag(self) -> np.ndarray | None:
        """单帧检测, 返回 T_cam_tag(pose6) 或 None。"""
        rgb, _ = self.camera.frame()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        dets = self._det.detect(gray, estimate_tag_pose=True,
                                camera_params=self._cam_params, tag_size=self.tag_size)
        if self.tag_id is not None:
            dets = [d for d in dets if d.tag_id == self.tag_id]
        if not dets:
            return None
        d = max(dets, key=lambda x: x.decision_margin)
        return Rp_to_pose(np.asarray(d.pose_R), np.asarray(d.pose_t).ravel())

    def locate_H(self, tcp_pose_world: np.ndarray, T_tag_hole: np.ndarray,
                 n_frames: int = 10) -> np.ndarray:
        """连拍 n 帧取平移中位数(姿态取最接近中位平移的那帧), 输出 {H} 世界系位姿。
        检出帧数不足一半时抛 RuntimeError。"""
        candidates = []
        for _ in range(n_frames):
            T_cam_tag = self.detect_tag()
            if T_cam_tag is not None:
                H = compose(compose(compose(tcp_pose_world, self.T_tcp_cam),
                                    T_cam_tag), np.asarray(T_tag_hole, float))
                candidates.append(H)
        if len(candidates) < n_frames // 2:
            raise RuntimeError(f"Tag 检出率不足: {len(candidates)}/{n_frames}")
        arr = np.array(candidates)
        med = np.median(arr[:, :3], axis=0)
        best = int(np.argmin(np.linalg.norm(arr[:, :3] - med, axis=1)))
        result = arr[best].copy()
        result[:3] = med
        return result
