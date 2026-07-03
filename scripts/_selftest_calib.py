"""标定求解算法离线自检（合成数据, 无需硬件）。通过后可删或保留作回归测试。"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calib.payload_id import G, solve_payload
from calib.dual_arm_calib import kabsch, rms_error

rng = np.random.default_rng(0)

# --- 1) 负载辨识: 已知真值合成 27 姿态测量(+20mN 噪声)后求解 ---
m_true, cog_true = 1.234, np.array([0.01, -0.02, 0.05])
bias_true = np.array([1.1, -2.2, 3.3, 0.11, -0.22, 0.33])
records = []
for _ in range(27):
    R = Rot.random(random_state=rng).as_matrix()
    fg = m_true * (R.T @ G)
    w = np.concatenate([bias_true[:3] + fg, bias_true[3:] + np.cross(cog_true, fg)])
    records.append((R, w + rng.normal(0, 0.02, 6)))
r = solve_payload(records)
assert abs(r["mass"] - m_true) < 0.01, r["mass"]
assert np.allclose(r["cog"], cog_true, atol=1e-3), r["cog"]
assert np.allclose(r["bias"], bias_true, atol=0.02), r["bias"]
print(f"payload solver OK: m={r['mass']:.4f} (true {m_true}), res_f_max={r['res_f_max']:.4f} N")

# --- 2) Kabsch: 已知刚体变换合成 6 对应点(+0.3mm 噪声) ---
R_true = Rot.from_euler("xyz", [5, -3, 170], degrees=True).as_matrix()
t_true = np.array([1.30, 0.05, -0.02])
Q = rng.uniform(-0.3, 0.3, (6, 3))
P = Q @ R_true.T + t_true + rng.normal(0, 0.0003, (6, 3))
R_est, t_est = kabsch(P, Q)
ang = np.degrees(np.arccos(np.clip((np.trace(R_est.T @ R_true) - 1) / 2, -1, 1)))
dt = np.linalg.norm(t_est - t_true)
assert ang < 0.2 and dt < 0.001, (ang, dt)
print(f"kabsch OK: ang_err={ang:.4f} deg, t_err={dt * 1000:.3f} mm, "
      f"RMS={rms_error(P, Q, R_est, t_est) * 1000:.3f} mm")

# --- 3) 挂日志后的模块导入检查 ---
from control.primitives import Primitives  # noqa: F401
import common.logger  # noqa: F401
print("all imports OK")
