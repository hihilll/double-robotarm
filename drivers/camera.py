"""RealSense 相机驱动（#29）。腕部 D405 / 全局 D435 共用。

接口约定: frame() 返回 (rgb, depth|None); intrinsics() 返回 {fx, fy, cx, cy}。
perception/ 只依赖此接口, 不依赖具体型号。
"""
from __future__ import annotations

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None  # type: ignore


class RealSenseCamera:
    def __init__(self, serial: str = "", width: int = 848, height: int = 480,
                 fps: int = 30, enable_depth: bool = True):
        if rs is None:
            raise RuntimeError("未安装 pyrealsense2: pip install pyrealsense2")
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        self._depth = enable_depth
        if enable_depth:
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            self.align = rs.align(rs.stream.color)   # 深度对齐到彩色
        profile = self.pipeline.start(cfg)
        vs = profile.get_stream(rs.stream.color).as_video_stream_profile()
        i = vs.get_intrinsics()
        self._intr = {"fx": i.fx, "fy": i.fy, "cx": i.ppx, "cy": i.ppy}
        self._depth_scale = (profile.get_device().first_depth_sensor()
                             .get_depth_scale() if enable_depth else None)
        for _ in range(10):   # 丢弃自动曝光稳定前的帧
            self.pipeline.wait_for_frames()

    def frame(self) -> tuple[np.ndarray, np.ndarray | None]:
        """返回 (rgb uint8 HxWx3, depth float32 米 HxW | None)。"""
        frames = self.pipeline.wait_for_frames()
        if self._depth:
            frames = self.align.process(frames)
        rgb = np.asanyarray(frames.get_color_frame().get_data())
        depth = None
        if self._depth:
            depth = (np.asanyarray(frames.get_depth_frame().get_data())
                     .astype(np.float32) * self._depth_scale)
        return rgb, depth

    def intrinsics(self) -> dict:
        return dict(self._intr)

    def close(self) -> None:
        self.pipeline.stop()
