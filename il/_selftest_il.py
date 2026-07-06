"""方案二离线自检（无硬件、无 lerobot 也能跑；装了 lerobot 则多测数据集创建）。

用法:  python il/_selftest_il.py
改 il/ 代码或 config_il.yaml 后重跑。⚠ 本自检不触碰方案一代码与配置文件内容,
仅只读加载验证接口联动。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- 1) 全模块可导入(硬件/学习库缺失时优雅降级) ---
import il.action_repr as ar                                          # noqa: E402
import il.adapters as ad                                             # noqa: E402
import il.teleop.spacemouse                                          # noqa: F401,E402
import il.teleop.teleop_loop                                         # noqa: F401,E402
import il.learning.cameras                                           # noqa: F401,E402
import il.learning.robot_dualarm as rd                               # noqa: E402
import il.learning.record_episodes                                   # noqa: F401,E402
import il.learning.convert_h5 as ch                                  # noqa: E402
import il.learning.train_compact as tc                               # noqa: E402
from il.learning.deploy_policy import TemporalEnsembler              # noqa: E402

print("imports OK (10 modules)")

# --- 2) 配置加载与键联动 ---
cfg = ad.load_il_config()
assert set(ad.MODE_CYCLE) <= set(cfg["stiffness_modes"]), "刚度档缺失"
for m in ad.MODE_CYCLE:
    k, kr = ad.stiffness_of_mode(cfg, m)
    assert k.shape == (3,) and kr.shape == (3,) and np.all(k > 0)
assert set(ch.STATE_TO_MODE.values()) <= set(cfg["stiffness_modes"])
for t in ("plug_insert", "plug_extract", "screw_place"):
    assert t in cfg["tasks"]
rc, ac = ad.load_configs()          # 方案一配置只读可加载
print("config wiring OK")

# --- 3) 刚度→导纳参数派生(临界阻尼) ---
k, kr = np.array([400.0, 100.0, 900.0]), np.array([10.0, 1.0, 40.0])
p = ad.params_from_stiffness(k, kr, cfg, ac["limits"])
a = cfg["admittance"]
d_expect = np.maximum(a["d_min"], 2 * a["zeta"] * np.sqrt(k * np.asarray(a["m"])))
assert np.allclose(p.d, d_expect), (p.d, d_expect)
assert np.allclose(p.k, k) and p.max_offset == ac["limits"]["max_offset"]
kc, krc = ad.clamp_stiffness([1e4, -5, 300], [100, 0, 10], cfg)
cl = cfg["deploy"]["stiffness_clamp"]
assert kc[0] == cl["k"][1] and kc[1] == cl["k"][0] and krc[0] == cl["kr"][1]
print("stiffness→admittance params OK")

# --- 4) 动作/状态编解码往返 ---
rng = np.random.default_rng(0)
for _ in range(50):
    pose = np.concatenate([rng.uniform(-1, 1, 3), rng.uniform(-2, 2, 3)])
    v = ar.pack_action(pose, k, kr, 0.37)
    assert v.shape == (ar.ACTION_DIM,) and v.dtype == np.float32
    out = ar.unpack_action(v)
    assert np.allclose(out["pose6"][:3], pose[:3], atol=1e-6)
    from scipy.spatial.transform import Rotation as Rot
    dR = (Rot.from_rotvec(out["pose6"][3:6]).inv() * Rot.from_rotvec(pose[3:6])).magnitude()
    assert dR < 1e-5, dR
    assert np.allclose(out["k"], k, atol=1e-3) and abs(out["gripper"] - 0.37) < 1e-6
s = ar.pack_state(pose, np.arange(6), 1.0)
assert s.shape == (ar.STATE_DIM,) and abs(s[-1] - 1.0) < 1e-9
R = ar.rot6d_to_matrix(rng.normal(size=6))       # 非正交输入 → Gram-Schmidt 仍为旋转阵
assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and abs(np.linalg.det(R) - 1) < 1e-9
assert len(ar.STATE_NAMES) == ar.STATE_DIM and len(ar.ACTION_NAMES) == ar.ACTION_DIM
print("action repr roundtrip OK")

# --- 5) temporal ensemble ---
ens = TemporalEnsembler(m=0.15, chunk_len=4)
assert ens.action_at(0.0) is None
c0 = np.tile(np.arange(4, dtype=float).reshape(4, 1), (1, 2))    # 块0@t=0: [0,1,2,3]
ens.add_chunk(0, c0)
assert np.allclose(ens.action_at(1.0), [1, 1])
assert np.allclose(ens.action_at(1.5), [1.5, 1.5])               # 步间线性插值
ens.add_chunk(2, c0 + 10)                                        # 块1@t=2, 重叠于步2/3
w = np.exp(-0.15 * np.arange(2))
expect = (2 * w[0] + 10 * w[1]) / w.sum()                        # 旧块权重更高
assert np.allclose(ens.action_at(2.0), [expect, expect])
assert np.allclose(ens.action_at(5.0), [13, 13])                 # 只剩块1覆盖
print("temporal ensemble OK")

# --- 6) H5→数据集转换(合成日志 + 桩数据集, 不需要 lerobot/真日志) ---
try:
    import h5py
except ImportError:
    h5py = None
if h5py is not None:
    with tempfile.TemporaryDirectory() as td:
        n = 250                                                   # 2s @125Hz
        path = Path(td) / "t.h5"
        with h5py.File(path, "w") as f:
            f["t"] = np.linspace(0, 2, n).reshape(-1, 1)
            f["state"] = np.repeat([[0], [1]], n // 2, 0)         # 前半 search 后半 insert
            f["tcp"] = np.tile(np.array([[.4, 0, .2, 0, 3.14, 0]]), (n, 1))
            f["target"] = f["tcp"][:]
            f["wrench_h"] = np.zeros((n, 6))
            f.attrs["state_codes"] = str({"search": 0, "insert": 1})

        class StubDS:                                             # 收集 add_frame 结果
            frames: list = []
            def add_frame(self, frame, task=None): self.frames.append(frame)
            def save_episode(self): pass

        ds = StubDS()
        n_out = ch.convert_file(path, ds, cfg, "test")
        assert n_out == len(ds.frames) == n // 5                  # 125→25Hz 下采
        k_mid, _ = ad.stiffness_of_mode(cfg, "mid")
        k_high, _ = ad.stiffness_of_mode(cfg, "high")
        assert np.allclose(ds.frames[0][rd.ACTION][9:12], k_mid)   # search→mid
        assert np.allclose(ds.frames[-1][rd.ACTION][9:12], k_high) # insert→high
    print("h5 converter OK (synthetic)")
else:
    print("h5 converter SKIP (无 h5py)")

# --- 7) 数据集 schema 一致性 + 训练命令组装 ---
feats = rd.dataset_features(cfg, ["wrist", "global"])
assert feats[rd.OBS_STATE]["shape"] == (ar.STATE_DIM,)
assert feats[rd.ACTION]["shape"] == (ar.ACTION_DIM,)
assert feats[rd.image_key("wrist")]["shape"] == (cfg["cameras"]["out_height"],
                                                 cfg["cameras"]["out_width"], 3)
cmd = tc.build_cmd("dualarm_test", cfg, Path("out"))
assert cmd[0] == "lerobot-train" and any("chunk_size=50" in c for c in cmd)
print("dataset schema + train cmd OK")

# --- 8) lerobot 可选检测 ---
try:
    import lerobot                                                # noqa: F401
    with tempfile.TemporaryDirectory() as td:
        cfg2 = {**cfg, "dataset": {**cfg["dataset"], "root": td}}
        ds = rd.create_or_resume_dataset("selftest_ds", cfg2, [])
        rd.add_frame_compat(ds, {rd.OBS_STATE: np.zeros(ar.STATE_DIM, np.float32),
                                 rd.ACTION: np.zeros(ar.ACTION_DIM, np.float32)}, "t")
        ds.save_episode()
    print("lerobot dataset create OK")
except ImportError:
    print("lerobot dataset SKIP (学习环境未装 lerobot——控制机上属正常)")

print("\nall il self-tests PASSED")
