"""任务#23: 125Hz 运行日志（§9 日志纪律）。

设计: 主环调用 log() 只做入队(微秒级, 绝不阻塞); 独立写线程批量落盘 HDF5。
通道 schema 由第一次 log() 的字段自动确定; 字符串 state 自动编码为整数
(映射存到文件属性)。数据格式对齐 LeRobot 迁移需求(二期 #36 直接写转换器)。

用法:
    logger = RunLogger("plug_insert")
    logger.log(wrench_h=w, target=target, tcp=pose, state="search")   # 每周期
    logger.close()
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore


class RunLogger:
    def __init__(self, run_name: str, out_dir: str | Path | None = None,
                 meta: dict | None = None):
        if h5py is None:
            raise RuntimeError("未安装 h5py: pip install h5py")
        out_dir = Path(out_dir or Path(__file__).resolve().parents[1] / "logs")
        out_dir.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = out_dir / f"{stamp}_{run_name}.h5"
        self._q: queue.Queue = queue.Queue(maxsize=10000)
        self._meta = meta or {}
        self._state_codes: dict[str, int] = {}
        self._t0 = time.monotonic()
        self._running = True
        self._thread = threading.Thread(target=self._writer, daemon=True, name="logger")
        self._thread.start()

    # ------------------------------------------------------------ 主环侧
    def log(self, state: str = "", **channels) -> None:
        """channels 值: float 或 一维 np.ndarray。满队列时丢帧(打印警告)而非阻塞主环。"""
        row = {"t": time.monotonic() - self._t0,
               "state": self._state_codes.setdefault(state, len(self._state_codes))}
        for k, v in channels.items():
            row[k] = np.asarray(v, dtype=np.float64).ravel()
        try:
            self._q.put_nowait(row)
        except queue.Full:
            print("[logger] 队列满, 丢帧(写盘过慢?)")

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=5.0)

    # ------------------------------------------------------------ 写线程侧
    def _writer(self) -> None:
        with h5py.File(self.path, "w") as f:
            dsets: dict[str, h5py.Dataset] = {}
            while self._running or not self._q.empty():
                batch: list[dict] = []
                try:
                    batch.append(self._q.get(timeout=0.2))
                    while len(batch) < 500:
                        batch.append(self._q.get_nowait())
                except queue.Empty:
                    pass
                if not batch:
                    continue
                for key in batch[0]:
                    col = np.stack([np.atleast_1d(r[key]) for r in batch])
                    if key not in dsets:
                        dsets[key] = f.create_dataset(
                            key, data=col, maxshape=(None,) + col.shape[1:],
                            chunks=(512,) + col.shape[1:], dtype="f8")
                    else:
                        d = dsets[key]
                        d.resize(d.shape[0] + col.shape[0], axis=0)
                        d[-col.shape[0]:] = col
            for k, v in self._meta.items():
                f.attrs[k] = str(v)
            f.attrs["state_codes"] = str(self._state_codes)
        print(f"[logger] 已保存 {self.path}")
