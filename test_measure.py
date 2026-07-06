"""measure パッケージのユニットテスト（TDD: RED → GREEN）"""
import numpy as np
import pytest
from pathlib import Path

GRD_PATH = Path(__file__).parent / "rei_1.grd"
GR2_PATH = Path(__file__).parent / "order made insole_rei1.gr2"


# ─────────────────────────── analysis ───────────────────────────

def test_left_right_ratio_equal_load():
    from measure.analysis import left_right_ratio
    grid = np.full((64, 16), -5.0)
    left, right = left_right_ratio(grid)
    assert abs(left - 50.0) < 0.01
    assert abs(right - 50.0) < 0.01


def test_left_right_ratio_sums_to_100():
    from measure.analysis import left_right_ratio
    rng = np.random.default_rng(42)
    grid = -rng.uniform(0.1, 20, (64, 16))
    left, right = left_right_ratio(grid)
    assert abs(left + right - 100.0) < 0.01


def test_front_back_ratio_sums_to_100():
    from measure.analysis import front_back_ratio
    rng = np.random.default_rng(0)
    foot = -rng.uniform(0.1, 20, (32, 16))
    front, back = front_back_ratio(foot)
    assert abs(front + back - 100.0) < 0.01


def test_contact_area_counts_negative_cells():
    from measure.analysis import contact_area
    foot = np.zeros((32, 16))
    foot[0, 0] = -1.0
    foot[1, 1] = -5.0
    assert contact_area(foot) == 2


def test_contact_area_zero_on_empty():
    from measure.analysis import contact_area
    assert contact_area(np.zeros((32, 16))) == 0


# ─────────────────────────── scanner ───────────────────────────

def test_stub_scanner_loads_grd():
    from measure.scanner import StubScanner
    scanner = StubScanner(str(GRD_PATH))
    assert scanner.is_connected()
    grid = scanner.scan()
    assert grid.shape == (64, 16)
    assert grid.dtype == np.float64


def test_stub_scanner_loads_gr2():
    from measure.scanner import StubScanner
    scanner = StubScanner(str(GR2_PATH))
    grid = scanner.scan()
    assert grid.shape == (64, 16)


def test_stub_scanner_scan_returns_negative_values():
    from measure.scanner import StubScanner
    scanner = StubScanner(str(GRD_PATH))
    grid = scanner.scan()
    assert grid.min() < 0


def test_stub_scanner_close():
    from measure.scanner import StubScanner
    scanner = StubScanner(str(GRD_PATH))
    scanner.close()
    assert not scanner.is_connected()


def test_stub_scanner_scan_returns_copy():
    """scan() は毎回独立したコピーを返す"""
    from measure.scanner import StubScanner
    scanner = StubScanner(str(GRD_PATH))
    g1 = scanner.scan()
    g1[0, 0] = 9999.0
    g2 = scanner.scan()
    assert g2[0, 0] != 9999.0


# ─────────────────────────── calibration ───────────────────────────

class _FixedScanner:
    """テスト用固定値スキャナー"""
    def __init__(self, value: float = -1.0):
        self._value = value

    def scan(self) -> np.ndarray:
        return np.full((64, 16), self._value)

    def is_connected(self) -> bool:
        return True

    def close(self) -> None:
        pass


def test_calibration_no_baseline_initially(tmp_path):
    from measure.calibration import CalibrationManager
    mgr = CalibrationManager(tmp_path / "baseline.npy")
    assert not mgr.has_baseline()


def test_calibration_capture_saves_file(tmp_path):
    from measure.calibration import CalibrationManager
    mgr = CalibrationManager(tmp_path / "baseline.npy")
    mgr.capture_baseline(_FixedScanner(-2.0))
    assert (tmp_path / "baseline.npy").exists()
    assert mgr.has_baseline()


def test_calibration_apply_subtracts_baseline(tmp_path):
    from measure.calibration import CalibrationManager
    mgr = CalibrationManager(tmp_path / "baseline.npy")
    mgr.capture_baseline(_FixedScanner(-1.0))  # baseline = -1.0

    raw = np.full((64, 16), -1.5)              # 0.5mm deeper → calibrated = -0.5 < -0.3 → kept
    result = mgr.apply(raw)
    assert np.allclose(result, -0.5)


def test_calibration_threshold_zeroes_shallow(tmp_path):
    from measure.calibration import CalibrationManager
    mgr = CalibrationManager(tmp_path / "baseline.npy")
    mgr.capture_baseline(_FixedScanner(-1.0))  # baseline = -1.0

    raw = np.full((64, 16), -1.1)   # calibrated = -0.1 > -0.3 → zeroed
    raw[0, 0] = -1.5                 # calibrated = -0.5 < -0.3 → kept
    result = mgr.apply(raw)
    assert result[1, 1] == 0.0
    assert result[0, 0] == pytest.approx(-0.5)


def test_calibration_same_scan_zeroes_out(tmp_path):
    """キャリブ後に同じデータを計測→ほぼ0"""
    from measure.calibration import CalibrationManager
    from measure.scanner import StubScanner
    mgr = CalibrationManager(tmp_path / "baseline.npy")
    scanner = StubScanner(str(GRD_PATH))
    mgr.capture_baseline(scanner)
    result = mgr.apply(scanner.scan())
    assert np.abs(result).max() < 1e-6


# ─────────────────────────── model ───────────────────────────

def test_session_stores_grids():
    from measure.model import MeasurementSession
    import grd_io
    raw = np.zeros((64, 16))
    patient = grd_io.PatientInfo(name="テスト")
    session = MeasurementSession(
        raw_grid=raw, calibrated_grid=raw.copy(), patient=patient, saved_path=None
    )
    assert session.patient.name == "テスト"
    assert session.timestamp is not None


def test_undo_stack_push_and_undo():
    from measure.model import UndoStack, MeasurementSession
    import grd_io
    stack = UndoStack()
    raw = np.zeros((64, 16))
    s1 = MeasurementSession(raw, raw.copy(), grd_io.PatientInfo(name="A"), None)
    s2 = MeasurementSession(raw, raw.copy(), grd_io.PatientInfo(name="B"), None)
    stack.push(s1)
    stack.push(s2)
    popped = stack.undo()
    assert popped is not None
    assert popped.patient.name == "A"


def test_undo_stack_max_2():
    """maxlen=2 なので最古エントリが脱落する"""
    from measure.model import UndoStack, MeasurementSession
    import grd_io
    stack = UndoStack()
    raw = np.zeros((64, 16))
    for name in ["A", "B", "C"]:
        stack.push(MeasurementSession(raw, raw.copy(), grd_io.PatientInfo(name=name), None))
    # stack = [B, C]  →  undo() で C を除去 → B を返す
    popped = stack.undo()
    assert popped is not None
    assert popped.patient.name == "B"


def test_undo_empty_stack_returns_none():
    from measure.model import UndoStack
    stack = UndoStack()
    assert stack.undo() is None


def test_undo_single_item_returns_none():
    """1件しかない場合は undo 不可"""
    from measure.model import UndoStack, MeasurementSession
    import grd_io
    stack = UndoStack()
    stack.push(MeasurementSession(
        np.zeros((64, 16)), np.zeros((64, 16)), grd_io.PatientInfo(), None
    ))
    assert stack.undo() is None
