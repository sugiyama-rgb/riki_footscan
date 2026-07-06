"""スキャナー抽象基底クラス・スタブ実装・シリアル実装・ワーカースレッド"""
from typing import Protocol
import time
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import grd_io

from PyQt6.QtCore import QThread, pyqtSignal

# GRDのスライス定数
RIGHT_SLICE = slice(0, 32)
LEFT_SLICE = slice(32, 64)


class ScannerBase(Protocol):
    def scan(self) -> np.ndarray: ...
    def is_connected(self) -> bool: ...
    def close(self) -> None: ...


class StubScanner:
    """GRD / GR2 ファイルを読み込み、常に同じグリッドを返すスタブ"""

    def __init__(self, path: str) -> None:
        self._data = grd_io.load(path)
        self._connected = True

    def scan(self) -> np.ndarray:
        return self._data.grid.copy()

    def is_connected(self) -> bool:
        return self._connected

    def close(self) -> None:
        self._connected = False


class SerialScanner:
    """実センサー（COM5 / 19200bps）からデータを取得するスキャナー

    プロトコル:
      送信: b'S' (0x53)
      受信: 1026 bytes = 513個の2バイトBE整数
            先頭512個 (インデックス0-511) = 32行×16列の圧力データ
            末尾1個 (インデックス512) はフレーム終端マーカー

    値の符号:
      センサー生値は足乗せで減少 (空: ~820, 足あり: ~650程度)
      scan()はそのまま正値で返す。
      CalibrationManager の (raw - baseline) が負値になり圧力を示す。

    左右配置:
      cols 0-7  = 左足
      cols 8-15 = 右足
    """

    BAUD = 19200
    SCAN_CMD = b'S'
    SENSOR_ROWS = 32
    SENSOR_COLS = 16
    _RESPONSE_BYTES = 1026

    def __init__(self, port: str) -> None:
        import serial
        self._ser = serial.Serial(port, self.BAUD, timeout=1)
        time.sleep(0.3)
        self._ser.reset_input_buffer()

    def scan(self) -> np.ndarray:
        """スキャン実行 → 32×16 float グリッドを返す（両足含む）

        低い値 = 接触あり（足乗せでADC値が低下する）
        """
        self._ser.reset_input_buffer()
        self._ser.write(self.SCAN_CMD)

        buf = bytearray()
        deadline = time.time() + 3.0
        while time.time() < deadline:
            n = self._ser.in_waiting
            if n > 0:
                buf.extend(self._ser.read(n))
                deadline = time.time() + 0.3
            else:
                time.sleep(0.05)
            if len(buf) >= self._RESPONSE_BYTES:
                break

        sensor_values = np.frombuffer(
            bytes(buf[:self.SENSOR_ROWS * self.SENSOR_COLS * 2]),
            dtype='>u2',
        ).astype(np.float32)
        return sensor_values.reshape(self.SENSOR_ROWS, self.SENSOR_COLS)

    def is_connected(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def close(self) -> None:
        if self._ser and self._ser.is_open:
            self._ser.close()


class ScanWorker(QThread):
    scan_done = pyqtSignal(object)   # np.ndarray
    scan_error = pyqtSignal(str)

    def __init__(self, scanner: ScannerBase) -> None:
        super().__init__()
        self._scanner = scanner

    def run(self) -> None:
        try:
            result = self._scanner.scan()
            self.scan_done.emit(result)
        except Exception as exc:
            self.scan_error.emit(str(exc))
