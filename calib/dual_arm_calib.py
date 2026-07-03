"""任务#19: 双臂基座标定——共触点法（§3.4）。

方法: 两臂各装尖头工具(TCP 已标到针尖, #18 先完成), 先后拖动(teachMode)触碰
同一组 ≥4 个不共面的空间点; 得到两基座系下的对应点集后, 用 SVD(Kabsch) 解
刚体变换 T_B5→B10, 使 p_B5 = R·p_B10 + t。输出 robots.yaml 片段并做留一交叉验证。

用法:  python calib/dual_arm_calib.py --points 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.task_frame import Rp_to_pose


# ------------------------------------------------------------ 纯求解(可离线自检)
def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """解 R,t 使 P ≈ R·Q + t。P,Q: (N,3) 对应点集。"""
    cp, cq = P.mean(axis=0), Q.mean(axis=0)
    H = (Q - cq).T @ (P - cp)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.linalg.det(Vt.T @ U.T)])   # 防反射
    R = Vt.T @ D @ U.T
    return R, cp - R @ cq


def rms_error(P: np.ndarray, Q: np.ndarray, R: np.ndarray, t: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((P - (Q @ R.T + t)) ** 2, axis=1))))


# ------------------------------------------------------------ 采集与主流程
def teach_point(arm, k: int) -> np.ndarray:
    arm.ctrl.teachMode()
    input(f"  拖动 [{arm.name}] 针尖触碰标定点 {k}, 保持接触, 回车记录...")
    arm.ctrl.endTeachMode()
    return arm.tcp_pose()[:3]


def main() -> None:
    import yaml
    from drivers.ur_arm import URArm

    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=5, help="标定点数(≥4, 不共面)")
    args = ap.parse_args()
    assert args.points >= 4

    cfg = yaml.safe_load((Path(__file__).parents[1] / "config" / "robots.yaml").read_text(encoding="utf-8"))
    ur5 = URArm("ur5", cfg["ur5"]["ip"], cfg["ur5"]["frequency"], cfg["ur5"]["tcp_offset"])
    ur10 = URArm("ur10", cfg["ur10"]["ip"], cfg["ur10"]["frequency"], cfg["ur10"]["tcp_offset"])
    ur5.connect(control=True)
    ur10.connect(control=True)

    print(f"共触点标定: {args.points} 个点, 布置在呈现区周围、彼此间距 >15cm 且不共面")
    P5, P10 = [], []
    try:
        for k in range(1, args.points + 1):
            print(f"--- 点 {k}/{args.points} ---")
            P5.append(teach_point(ur5, k))
            P10.append(teach_point(ur10, k))
    finally:
        ur5.disconnect()
        ur10.disconnect()

    P5, P10 = np.array(P5), np.array(P10)
    R, t = kabsch(P5, P10)
    rms = rms_error(P5, P10, R, t)

    # 留一交叉验证: 每次去掉一个点重解, 用被去掉的点评估
    loo = []
    for i in range(len(P5)):
        mask = np.arange(len(P5)) != i
        Ri, ti = kabsch(P5[mask], P10[mask])
        loo.append(np.linalg.norm(P5[i] - (Ri @ P10[i] + ti)))

    print(f"\n拟合 RMS = {rms * 1000:.2f} mm, 留一验证最大偏差 = {max(loo) * 1000:.2f} mm "
          f"→ {'达标 ✓' if max(loo) < 0.0015 else '未达标 ✗(检查针尖TCP标定/触点是否打滑)'}")
    pose = Rp_to_pose(R, t)
    print(f"\n粘贴到 robots.yaml:\ndual_arm_calib:\n"
          f"  pose_b10_in_world: [{', '.join(f'{v:.6f}' for v in pose)}]")


if __name__ == "__main__":
    main()
