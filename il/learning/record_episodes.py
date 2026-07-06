"""演示数据采集入口（任务#43/#44）。

流程(每条 episode):
  [UR10 随机呈现] → 回车开始 → 控制线程 125Hz 跑 TeleopSession
  → 主线程 25Hz 组帧写 LeRobotDataset → 回车结束 → s=保存 / d=废弃
观测/动作 schema 见 il/action_repr.py; 帧率与对齐参数在 config_il.yaml。

用法:
  python il/learning/record_episodes.py --task plug_insert --episodes 10
  python il/learning/record_episodes.py --task plug_insert --no-ur10   # 桌面台架
⚠ 采集纪律(#43): 相机位/TCP/传感器安装一经开采不得变动; 变动 = 数据作废。
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from il import action_repr as ar
from il.adapters import load_il_config
from il.learning.robot_dualarm import (ACTION, OBS_STATE, DualArmRig,
                                       add_frame_compat, create_or_resume_dataset,
                                       image_key)
from il.teleop.teleop_loop import TeleopSession


def record_one(session: TeleopSession, rig: DualArmRig, ds, task_label: str,
               fps: float) -> int:
    """录制单条。返回帧数。调用方决定保存/废弃。"""
    stop = threading.Event()
    ctrl = threading.Thread(target=session.run, args=(stop.is_set,), daemon=True)
    session.reset()
    ctrl.start()

    waiter = threading.Thread(target=lambda: (input(), stop.set()), daemon=True)
    waiter.start()                                 # 回车 → 结束本条
    n, dt = 0, 1.0 / fps
    next_t = time.monotonic()
    while not stop.is_set() and session.running:
        next_t += dt
        snap, age = session.snapshot.get()
        if snap is None:
            time.sleep(0.01)
            continue
        frame = {
            OBS_STATE: ar.pack_state(snap["tcp"], snap["wrench"], snap["grip"]),
            ACTION: ar.pack_action(snap["ref"], snap["k"], snap["kr"], snap["grip"]),
        }
        if rig.cameras is not None:
            for name, img in rig.cameras.frames().items():
                frame[image_key(name)] = img
        add_frame_compat(ds, frame, task_label)
        n += 1
        time.sleep(max(0.0, next_t - time.monotonic()))
    stop.set()
    ctrl.join(timeout=2.0)
    if session.error:
        print(f"  本条发生安全停止({session.error}), 建议废弃")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="plug_insert / plug_extract / screw_place")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--payload", default=None, help="默认按任务映射(插入=plug 等)")
    ap.add_argument("--no-ur10", action="store_true", help="桌面台架: 不接 UR10")
    ap.add_argument("--no-cameras", action="store_true", help="管线调试: 只录状态/动作")
    args = ap.parse_args()

    il_cfg = load_il_config()
    payload = args.payload or {"plug_insert": "plug", "plug_extract": "none",
                               "screw_place": "screw"}[args.task]
    rig = DualArmRig(il_cfg, payload=payload, use_ur10=not args.no_ur10,
                     use_cameras=not args.no_cameras)
    cam_names = rig.cameras.names if rig.cameras is not None else []
    repo_id = f"{il_cfg['dataset']['repo_prefix']}_{args.task}"
    ds = create_or_resume_dataset(repo_id, il_cfg, cam_names)
    session = TeleopSession(rig.stack, il_cfg, gripper=rig.gripper)
    rng = np.random.default_rng()
    task_label = f"dual-arm {args.task.replace('_', ' ')}"

    saved = 0
    print(f"采集 {args.episodes} 条 [{args.task}]。每条: 回车开始→演示→回车结束→s/d")
    print("⚠ 速度滑块 30%; SpaceMouse 键0=刚度档 键1=夹爪; 演示节奏刻意多样化")
    try:
        while saved < args.episodes:
            if rig.ur10 is not None:
                p = rig.present_ur10(args.task, rng)
                print(f"  UR10 已呈现(含扰动): {np.round(p, 4)}")
            input(f"--- 第 {saved + 1}/{args.episodes} 条: 回车开始录制 ---")
            n = record_one(session, rig, ds, task_label, il_cfg["freq_record"])
            ans = input(f"  {n} 帧 ({n / il_cfg['freq_record']:.1f}s)。s=保存 d=废弃: ").strip()
            if ans.lower() == "s" and n > 0 and not session.error:
                ds.save_episode()
                saved += 1
                print(f"  ✓ 已存 (总 {saved})")
            else:
                ds.clear_episode_buffer()
                print("  ✗ 已废弃")
    finally:
        session.close()
        rig.close()
    print(f"\n完成: {saved} 条 → {il_cfg['dataset']['root']}/{repo_id}")


if __name__ == "__main__":
    main()
