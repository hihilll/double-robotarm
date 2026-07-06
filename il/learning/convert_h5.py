"""方案一 HDF5 运行日志 → LeRobotDataset v3.0 转换器（任务#38, 兼收原#36）。

用途:
  1) 无硬件阶段的训练管线冒烟测试（状态-动作数据, 无图像通道）;
  2) 方案一真机日志迁移为学习格式。
映射:
  observation.state ← tcp(位姿) + wrench_h + gripper(日志无夹爪, 置常数1)
  action            ← target(位姿) + 按状态机标签映射的刚度档 + gripper
  125Hz → fps 下采样(每 5 帧取 1)
状态→刚度档映射: 方案一日志只有模式化导纳, 用档位近似(见 STATE_TO_MODE)。

用法:  python il/learning/convert_h5.py --h5 logs/*.h5 --repo dualarm_h5_smoke
"""
from __future__ import annotations

import argparse
import ast
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from il import action_repr as ar
from il.adapters import load_il_config, stiffness_of_mode
from il.learning.robot_dualarm import (ACTION, OBS_STATE, add_frame_compat,
                                       create_or_resume_dataset)

# 方案一状态机标签 → 刚度档（近似标注; ""=空标签按 low）
STATE_TO_MODE = {"guarded_move": "mid", "search": "mid", "insert": "high",
                 "extract": "high", "": "low"}


def convert_file(path: Path, ds, il_cfg: dict, task_label: str) -> int:
    import h5py
    stride = int(round(125.0 / il_cfg["dataset"]["fps"]))
    with h5py.File(path, "r") as f:
        codes = {v: k for k, v in ast.literal_eval(f.attrs["state_codes"]).items()}
        state = f["state"][:, 0].astype(int)
        tcp = f["tcp"][:]
        target = f["target"][:]
        wrench = f["wrench_h"][:]
    n = 0
    for i in range(0, len(state), stride):
        mode = STATE_TO_MODE.get(codes.get(state[i], ""), "low")
        k, kr = stiffness_of_mode(il_cfg, mode)
        frame = {
            OBS_STATE: ar.pack_state(tcp[i], wrench[i], 1.0),
            ACTION: ar.pack_action(target[i], k, kr, 1.0),
        }
        add_frame_compat(ds, frame, task_label)
        n += 1
    ds.save_episode()
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", nargs="+", required=True, help="HDF5 日志路径(可通配)")
    ap.add_argument("--repo", required=True, help="输出数据集名")
    ap.add_argument("--task-label", default="plan1 log replay")
    args = ap.parse_args()

    il_cfg = load_il_config()
    ds = create_or_resume_dataset(args.repo, il_cfg, cam_names=[])  # 无图像通道
    files = [Path(p) for pat in args.h5 for p in sorted(glob.glob(pat))]
    assert files, f"未匹配到文件: {args.h5}"
    total = 0
    for p in files:
        n = convert_file(p, ds, il_cfg, args.task_label)
        total += n
        print(f"  {p.name} → {n} 帧")
    print(f"\n转换完成: {len(files)} 条 episode / {total} 帧 → {args.repo}")


if __name__ == "__main__":
    main()
