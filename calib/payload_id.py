"""任务#16: 负载辨识与重力补偿参数求解（§3.2）。

方法: 自动遍历腕部姿态(q4/q5/q6 组合, 27 姿态), 每姿态静止 1s 取 F/T 均值;
模型对未知量线性, 两段最小二乘:
  第一段  F_i = m·g_s_i + b_F          → 解 [m, b_F]   (g_s_i = R_iᵀ·g)
  第二段  T_i = cog×(m·g_s_i) + b_T    → 解 [cog, b_T]  (= −skew(f_g_i)·cog + b_T)
输出可直接粘贴进 robots.yaml payloads 的 yaml 片段。

⚠ 运行前: 夹爪已夹持对应工件; 末端周围留出 >0.4m 空间; 速度滑块 30%。
用法:  python calib/payload_id.py --payload plug
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.task_frame import pose_to_Rp

G = np.array([0.0, 0.0, -9.81])


def skew(a: np.ndarray) -> np.ndarray:
    return np.array([[0, -a[2], a[1]],
                     [a[2], 0, -a[0]],
                     [-a[1], a[0], 0]], dtype=float)


# ------------------------------------------------------------ 纯求解(可离线自检)
def solve_payload(records: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """records: [(R_base_sensor 3x3, wrench 6), ...]。返回 m/cog/bias 及残差。"""
    # 第一段: 力方程解 m 和 b_F
    A1 = np.vstack([np.hstack([(R.T @ G).reshape(3, 1), np.eye(3)]) for R, _ in records])
    y1 = np.concatenate([w[:3] for _, w in records])
    x1, *_ = np.linalg.lstsq(A1, y1, rcond=None)
    m, b_f = float(x1[0]), x1[1:4]

    # 第二段: 力矩方程解 cog 和 b_T
    A2 = np.vstack([np.hstack([-skew(m * (R.T @ G)), np.eye(3)]) for R, _ in records])
    y2 = np.concatenate([w[3:6] for _, w in records])
    x2, *_ = np.linalg.lstsq(A2, y2, rcond=None)
    cog, b_t = x2[:3], x2[3:6]

    # 残差(逐姿态补偿后应接近零)
    res_f, res_t = [], []
    for R, w in records:
        f_g = m * (R.T @ G)
        res_f.append(np.linalg.norm(w[:3] - b_f - f_g))
        res_t.append(np.linalg.norm(w[3:6] - b_t - np.cross(cog, f_g)))
    return {"mass": m, "cog": cog, "bias": np.concatenate([b_f, b_t]),
            "res_f_max": float(np.max(res_f)), "res_t_max": float(np.max(res_t))}


# ------------------------------------------------------------ 采集与主流程
def collect(arm, ft, R_tcp_sensor: np.ndarray, offsets_deg=(-60, 0, 60),
            settle: float = 1.0, n_avg: int = 50) -> list:
    q0 = arm.joints()
    records = []
    combos = list(product(offsets_deg, repeat=3))
    for i, (d4, d5, d6) in enumerate(combos):
        q = q0.copy()
        q[3:6] += np.radians([d4, d5, d6])
        arm.ctrl.moveJ(q.tolist(), 0.8, 1.2)
        time.sleep(settle)
        samples = []
        for _ in range(n_avg):
            w, _ = ft.latest()
            samples.append(w)
            time.sleep(0.01)
        R_base_tcp, _ = pose_to_Rp(arm.tcp_pose())
        records.append((R_base_tcp @ R_tcp_sensor, np.mean(samples, axis=0)))
        print(f"  姿态 {i + 1}/{len(combos)} 采集完成")
    arm.ctrl.moveJ(q0.tolist(), 0.8, 1.2)
    return records


def main() -> None:
    import yaml
    from drivers.ft_sensor import FTReader
    from drivers.ur_arm import URArm

    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", required=True, help="工件名(robots.yaml payloads 键): none/plug/screw")
    args = ap.parse_args()

    cfg = yaml.safe_load((Path(__file__).parents[1] / "config" / "robots.yaml").read_text(encoding="utf-8"))
    fc = cfg["ft_sensor"]
    arm = URArm("ur5", cfg["ur5"]["ip"], cfg["ur5"]["frequency"], cfg["ur5"]["tcp_offset"])
    arm.connect(control=True)
    ft = FTReader(fc["ip"], fc["port"], fc["counts_per_force"],
                  fc["counts_per_torque"], lpf_cutoff_hz=5.0)  # 静态采集用更低截止
    ft.start()

    input(f"确认已夹持 [{args.payload}] 且末端周围 >0.4m 无障碍, 回车开始(27 姿态约 3 分钟)...")
    try:
        records = collect(arm, ft, np.asarray(fc["R_tcp_sensor"], float))
    finally:
        ft.stop()
        arm.disconnect()

    r = solve_payload(records)
    ok = r["res_f_max"] < 0.5 and r["res_t_max"] < 0.05
    print(f"\n辨识结果: m={r['mass']:.4f} kg, 残差 |F|max={r['res_f_max']:.3f} N "
          f"|T|max={r['res_t_max']:.4f} Nm  → {'达标 ✓' if ok else '未达标 ✗(检查夹持松动/传感器漂移)'}")
    print(f"\n粘贴到 robots.yaml payloads:\n"
          f"  {args.payload}: {{mass: {r['mass']:.4f}, "
          f"cog: [{', '.join(f'{v:.5f}' for v in r['cog'])}], "
          f"bias: [{', '.join(f'{v:.4f}' for v in r['bias'])}]}}")


if __name__ == "__main__":
    main()
