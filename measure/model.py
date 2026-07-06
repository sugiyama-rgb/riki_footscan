"""計測セッションデータモデルと Undo スタック"""
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import grd_io


@dataclass
class MeasurementSession:
    raw_grid: np.ndarray
    calibrated_grid: np.ndarray
    patient: grd_io.PatientInfo
    saved_path: Path | None
    timestamp: datetime = field(default_factory=datetime.now)


class UndoStack:
    def __init__(self) -> None:
        self._stack: deque[MeasurementSession] = deque(maxlen=2)

    def push(self, session: MeasurementSession) -> None:
        self._stack.append(session)

    def undo(self) -> MeasurementSession | None:
        if len(self._stack) < 2:
            return None
        self._stack.pop()          # 現在の状態を除去
        return self._stack[-1]     # ひとつ前の状態を返す
