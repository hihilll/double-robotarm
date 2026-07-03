"""任务#20: eye-in-hand 手眼标定（§3.3）。

方法: 拖动示教(teachMode) UR5 到 15–20 个观察固定 AprilTag 板的位姿(平移+旋转
都要有变化, 倾角差 >30°), 每姿态记录 (T_base_tcp, T_cam_tag);
cv2.calibrateHandEye(Tsai 法) 解 T_tcp_cam。用"标签在基座系位置的散布"做验证。

用法:  python calib/handeye.py --poses 15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.task_frame import Rp_to_pose, compose, pose_to_Rp


# ------------------------------------------------------------ 纯求解(可离线自检)
def solve_handeye(tcp_poses: list[np.ndarray], tag_poses: list[np.ndarray]) -> np.ndarray:
    """tcp_poses: T_base_tcp 列表; tag_poses: T_cam_tag 列表。返回 T_tcp_cam(pose6)。"""
    import cv2
    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    for tcp, tag in zip(tcp_poses, tag_poses):
        R, p = pose_to_Rp(tcp)
        R_g2b.append(R); t_g2b.append(p)
        R, p = pose_to_Rp(tag)
        R_t2c.append(R); t_t2c.append(p)
    R_c2g, t_c2g = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c,
                                        method=cv2.CALIB_HAND_EYE_TSAI)
    return Rp_to_pose(R_c2g, t_c2g.ravel())


def validate(tcp_poses, tag_poses, T_tcp_cam) -> float:
    """标签是固定的 ⇒ 各样本算出的 T_base_tag 平移散布 = 标定误差(m)。"""
    pts = np.array([compose(compose(tcp, T_tcp_cam), tag)[:3]
                    for tcp, tag in zip(tcp_poses, tag_poses)])
    return float(np.linalg.norm(pts - pts.mean(axis=0), axis=1).max())


# ------------------------------------------------------------ 主流程
def main() -> None:
    from common.setup import load_configs
    from drivers.camera import RealSenseCamera
    from drivers.ur_arm import URArm
    from perception.tag_locator import TagLocator

    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", type=int, default=15)
    args = ap.parse_args()

    rc, _ = load_configs()
    cam_cfg = rc["camera"]
    arm = URArm("ur5", rc["ur5"]["ip"], rc["ur5"]["frequency"], rc["ur5"]["tcp_offset"])
    arm.connect(control=True)
    camera = RealSenseCamera(**cam_cfg["wrist"], enable_depth=False)
    # 手眼标定阶段 T_tcp_cam 未知, 传单位位姿, 只用 detect_tag()
    loc = TagLocator(camera, cam_cfg["tag"]["family"], cam_cfg["tag"]["size"],
                     np.zeros(6))

    print(f"固定 AprilTag 标定板于桌面。采集 {args.poses} 个观察位姿, "
          f"注意平移和倾角都要有明显变化。")
    tcp_poses, tag_poses = [], []
    try:
        k = 0
        while k < args.poses:
            arm.ctrl.teachMode()
            input(f"  [{k + 1}/{args.poses}] 拖动到新观察位姿(标签在视野内), 回车记录...")
            arm.ctrl.endTeachMode()
            samples = [t for t in (loc.detect_tag() for _ in range(5)) if t is not None]
            if len(samples) < 3:
                print("    ✗ 检出不足, 换位姿重试")
                continue
            tcp_poses.append(arm.tcp_pose())
            tag_poses.append(np.median(np.array(samples), axis=0))
            k += 1
    finally:
        camera.close()
        arm.disconnect()

    T_tcp_cam = solve_handeye(tcp_poses, tag_poses)
    err = validate(tcp_poses, tag_poses, T_tcp_cam)
    print(f"\n验证: 标签基座系位置最大散布 = {err * 1000:.2f} mm "
          f"→ {'达标 ✓' if err < 0.0015 else '未达标 ✗(增加位姿多样性/检查tag尺寸)'}")
    print(f"\n粘贴到 robots.yaml camera:\n"
          f"  T_tcp_cam: [{', '.join(f'{v:.6f}' for v in T_tcp_cam)}]")


if __name__ == "__main__":
    main()
