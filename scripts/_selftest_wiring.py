"""全模块导入 + 配置/代码键名联动自检（无需硬件）。改配置或任务代码后重跑。"""
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]

# --- 1) 全模块可导入(硬件库缺失时应优雅降级而非 ImportError) ---
import drivers.bus, drivers.ur_arm, drivers.ft_sensor, drivers.gripper, drivers.camera  # noqa
import control.task_frame, control.admittance, control.safety, control.primitives      # noqa
import common.logger, common.setup                                                     # noqa
import perception.tag_locator                                                          # noqa
import calib.payload_id, calib.dual_arm_calib, calib.handeye                           # noqa
from tasks.insertion import InsertionTask
from tasks.plug_extract import PlugExtractTask
from tasks.dual_arm_manager import PAYLOAD_OF, DualArmManager  # noqa
print("imports OK (16 modules)")

# --- 2) 配置键名与代码取值联动 ---
rc = yaml.safe_load((ROOT / "config" / "robots.yaml").read_text(encoding="utf-8"))
ac = yaml.safe_load((ROOT / "config" / "admittance.yaml").read_text(encoding="utf-8"))

from control.admittance import AdmittanceParams
for mode in ("hand_push_demo", "search", "insert"):
    AdmittanceParams.from_dict(ac["modes"][mode], ac["limits"])
from control.safety import SafetyLimits
SafetyLimits.from_dict(ac["safety"])

# InsertionTask.run 用到的键
need_insert = {"contact_force", "search_force", "spiral_pitch", "spiral_v",
               "spiral_r_max", "drop_dz", "insert_force", "insert_depth"}
for t in ("plug_insert", "screw_place"):
    missing = need_insert - set(ac["tasks"][t])
    assert not missing, f"{t} 缺: {missing}"
# PlugExtractTask 用到的键
need_ext = {"grasp_advance", "f_start", "f_rate", "f_max", "extract_dist", "wiggle_above"}
assert not (need_ext - set(ac["tasks"]["plug_extract"]))
# 状态机与管理器用到的键
need_sm = {"standoff", "approach_v", "contact_v", "retries", "vision_consistency"}
assert not (need_sm - set(ac["state_machine"]))
for t in PAYLOAD_OF:
    assert t in rc["task_geometry"], f"robots.yaml task_geometry 缺 {t}"
    assert t in rc["camera"]["tag"]["T_tag_hole"], f"T_tag_hole 缺 {t}"
    assert rc["payloads"].get(PAYLOAD_OF[t]) is not None
assert {"family", "size"} <= set(rc["camera"]["tag"])
print("config wiring OK")

# --- 3) 任务几何链路演算: 合成 UR10 位姿 → {H} → entry_pose 应在孔口上方 ---
from control.task_frame import TaskFrame, compose, pose_to_Rp
T_w_b10 = np.asarray(rc["dual_arm_calib"]["pose_b10_in_world"], float)
tcp10 = np.array([0.5, 0.0, 0.4, 0.0, 3.1416, 0.0])
T_tcp_hole = np.asarray(rc["task_geometry"]["plug_insert"]["hole_in_ur10_tcp"], float)
H_pose = compose(compose(T_w_b10, tcp10), T_tcp_hole)
H = TaskFrame(H_pose)
entry = H.entry_pose(0.015)
gap = H.p - entry[:3]
assert abs(np.linalg.norm(gap) - 0.015) < 1e-9          # 距孔口 15mm
assert np.allclose(gap / 0.015, H.R[:, 2], atol=1e-6)   # 沿 -Z_H 方向后退
print(f"geometry chain OK: H at {np.round(H.p, 3)}, entry standoff verified")
print("\nall wiring self-tests PASSED")
