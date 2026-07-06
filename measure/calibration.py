"""キャリブレーション管理（ベースライン差分＋閾値処理）"""
from pathlib import Path

import numpy as np
from platformdirs import user_config_dir

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from measure.scanner import ScannerBase

THRESHOLD_MM = -0.3   # これより浅い接触（> -0.3）はノイズとして0にする
_ADC_SCALE = 20.0 / 600.0  # ADC差分 → mm相当値換算係数（ADC600単位 ≈ 20mm）


def _default_path() -> Path:
    return Path(user_config_dir("riki_measure")) / "baseline.npy"


class CalibrationManager:
    def __init__(self, baseline_path: Path | None = None) -> None:
        self._path = baseline_path if baseline_path is not None else _default_path()
        self._baseline: np.ndarray | None = None
        if self._path.exists():
            self._baseline = np.load(self._path)

    def capture_baseline(self, scanner: ScannerBase) -> None:
        self._baseline = scanner.scan()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self._path, self._baseline)

    def apply(self, raw: np.ndarray) -> np.ndarray:
        if self._baseline is None or self._baseline.shape != raw.shape:
            return raw.copy()
        # ADC値は足乗せで低下するため raw - baseline は負値 → mm換算
        calibrated = (raw - self._baseline) * _ADC_SCALE
        calibrated[calibrated > THRESHOLD_MM] = 0.0
        return calibrated

    def has_baseline(self) -> bool:
        return self._baseline is not None

    def reset_baseline(self) -> None:
        self._baseline = None
        if self._path.exists():
            self._path.unlink()
