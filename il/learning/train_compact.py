"""训练入口（任务#40/#45）：组装 lerobot-train 命令并执行。

Comp-ACT 的落地方式(方案二 D1/D2): 刚度已并入 16 维动作向量, 对 lerobot 的
ACT 就是普通回归维度——零框架改动, 直接用官方训练器。
两组对照(#45):
  ① baseline ACT: 先用 il/tools 把数据集动作截为 10 维(位姿+夹爪)再训 —— 或
     直接训 16 维但部署时忽略刚度维、用固定档(更简单, 推荐);
  ② Comp-ACT: 16 维全动作。
若 lerobot 升级导致参数名变化: 跑 `lerobot-train --help` 核对后改本文件 CLI_ARGS。

用法:
  python il/learning/train_compact.py --repo dualarm_plug_insert            # 执行
  python il/learning/train_compact.py --repo dualarm_plug_insert --dry-run  # 只打印
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from il.adapters import ROOT, load_il_config


def build_cmd(repo: str, il_cfg: dict, out_dir: Path, resume: bool = False) -> list[str]:
    t = il_cfg["train"]
    root = ROOT / il_cfg["dataset"]["root"] / repo
    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={repo}",
        f"--dataset.root={root}",
        f"--policy.type={t['policy']}",
        f"--policy.chunk_size={t['chunk_size']}",
        f"--policy.n_action_steps={t['n_action_steps']}",
        f"--policy.device={t['device']}",
        f"--batch_size={t['batch_size']}",
        f"--steps={t['steps']}",
        f"--policy.optimizer_lr={t['lr']}",
        f"--output_dir={out_dir}",
        f"--job_name={repo}",
        "--policy.push_to_hub=false",
        "--wandb.enable=false",
    ]
    if resume:
        cmd.append("--resume=true")
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="数据集名(dataset.root 下目录名)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    il_cfg = load_il_config()
    out_dir = ROOT / "il" / "logs_il" / "train" / f"{args.repo}_{time.strftime('%m%d_%H%M')}"
    cmd = build_cmd(args.repo, il_cfg, out_dir, args.resume)
    print("训练命令:\n  " + " \\\n  ".join(cmd))
    if args.dry_run:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    raise SystemExit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
