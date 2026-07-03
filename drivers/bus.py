"""线程间共享数据：最新值覆盖缓冲。

力控环永远读"最新一帧"，绝不排队等待——这是 §1.4 通讯架构的核心约定。
"""
from __future__ import annotations

import threading
import time
from typing import Any, Optional, Tuple


class LatestValue:
    """线程安全的单槽缓冲：写者覆盖，读者取最新值及其时间戳。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: Any = None
        self._stamp: float = 0.0

    def put(self, value: Any) -> None:
        with self._lock:
            self._value = value
            self._stamp = time.monotonic()

    def get(self) -> Tuple[Optional[Any], float]:
        """返回 (value, 数据年龄秒)。value 为 None 表示从未写入。"""
        with self._lock:
            if self._value is None:
                return None, float("inf")
            return self._value, time.monotonic() - self._stamp
