"""日志回放绘图（#23 配套）: 力曲线 + Z 深度 + 状态机时间线。

用法:  python scripts/plot_log.py logs/20260715_103000_plug_insert.h5
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    path = Path(sys.argv[1])
    with h5py.File(path, "r") as f:
        t = f["t"][:, 0]
        state = f["state"][:, 0].astype(int)
        codes = {v: k for k, v in ast.literal_eval(f.attrs["state_codes"]).items()}
        w = f["wrench_h"][:] if "wrench_h" in f else None
        tcp = f["tcp"][:] if "tcp" in f else None

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 9))
    if w is not None:
        for i, name in enumerate(["Fx", "Fy", "Fz"]):
            axes[0].plot(t, w[:, i], label=name)
        axes[0].set_ylabel("F (N)")
        axes[0].legend(); axes[0].grid(True)
        for i, name in enumerate(["Tx", "Ty", "Tz"]):
            axes[1].plot(t, w[:, 3 + i], label=name)
        axes[1].set_ylabel("T (Nm)")
        axes[1].legend(); axes[1].grid(True)
    if tcp is not None:
        ax2 = axes[2].twinx()
        ax2.plot(t, tcp[:, 2] * 1000, "k", alpha=0.5)
        ax2.set_ylabel("TCP z (mm)")
    axes[2].step(t, state, where="post")
    axes[2].set_yticks(sorted(codes), [codes[c] for c in sorted(codes)])
    axes[2].set_xlabel("t (s)"); axes[2].grid(True)
    fig.suptitle(path.name)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
