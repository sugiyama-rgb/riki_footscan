"""アーチ調整（持ち上げ／へこませる＝免荷）機能のユニットテスト（TDD: RED → GREEN）"""
import numpy as np
import pytest

import grd_io
from foot_model import ARCH_MAX_LOWER_MM, ArchParams, FootModel, preview_arch_max


def _make_model(baseline_mm: float = -10.0) -> FootModel:
    """全セルに既存スキャンデータ相当の負の深さ(baseline_mm)を持つモデルを作る。
    0埋めのグリッドだとnp.minimum(x, 0.0)クランプで常に0になってしまい、
    持ち上げ/へこませ量の検証ができないため、非ゼロのベースラインを使う。"""
    grid = np.full((grd_io.GRID_ROWS, grd_io.GRID_COLS), baseline_mm, dtype=np.float64)
    grd = grd_io.GrdData(grid=grid, patient=grd_io.PatientInfo(), raw_meta_lines=[])
    return FootModel(grd)


def _rect_mask(r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    mask = np.zeros((32, 16), dtype=bool)
    mask[r0:r1, c0:c1] = True
    return mask


# ─────────────────────── へこませる方向（免荷） ───────────────────────

def test_negative_height_mm_lowers_cells_inside_mask():
    model = _make_model(baseline_mm=-5.0)
    mask = _rect_mask(10, 20, 4, 12)
    params = ArchParams(mask=mask, height_mm=-3.0, smoothing=0.5)
    model.apply_arch("right", params)

    assert model.right_grid[15, 8] < -5.0


def test_lower_direction_never_exceeds_base_minus_max_lower_mm():
    model = _make_model(baseline_mm=-5.0)
    mask = _rect_mask(10, 20, 4, 12)
    params = ArchParams(mask=mask, height_mm=-50.0, smoothing=0.5)
    model.apply_arch("right", params)

    floor = -5.0 - ARCH_MAX_LOWER_MM
    assert (model.right_grid >= floor - 1e-6).all()
    assert model.right_grid[15, 8] == pytest.approx(floor, abs=0.5)


def test_repeated_lower_applications_still_respect_the_floor():
    model = _make_model(baseline_mm=-5.0)
    mask = _rect_mask(10, 20, 4, 12)
    for _ in range(5):
        model.apply_arch("right", ArchParams(mask=mask, height_mm=-8.0, smoothing=0.5))

    floor = -5.0 - ARCH_MAX_LOWER_MM
    assert (model.right_grid >= floor - 1e-6).all()


def test_apply_arch_stats_report_negative_extreme_for_lower_direction():
    model = _make_model(baseline_mm=-5.0)
    mask = _rect_mask(10, 20, 4, 12)
    stats = model.apply_arch("right", ArchParams(mask=mask, height_mm=-3.0, smoothing=0.5))

    assert stats["actual_max"] < 0
    assert stats["affected"] > 0


# ─────────────────────── 持ち上げ方向（既存挙動の回帰） ───────────────────────

def test_raise_direction_still_clamps_at_zero_with_floor_logic_present():
    model = _make_model(baseline_mm=-10.0)
    mask = _rect_mask(10, 20, 4, 12)
    params = ArchParams(mask=mask, height_mm=50.0, smoothing=0.5)
    model.apply_arch("right", params)

    assert (model.right_grid <= 0.0).all()
    assert model.right_grid[15, 8] == pytest.approx(0.0, abs=0.01)


def test_apply_arch_stats_report_positive_extreme_for_raise_direction():
    model = _make_model(baseline_mm=-10.0)
    mask = _rect_mask(10, 20, 4, 12)
    stats = model.apply_arch("right", ArchParams(mask=mask, height_mm=3.0, smoothing=0.5))

    assert stats["actual_max"] > 0
    assert stats["affected"] > 0


# ─────────────────────── プレビュー ───────────────────────

def test_preview_arch_max_is_negative_for_lower_direction():
    mask = _rect_mask(5, 25, 2, 14)
    preview = preview_arch_max(mask, height_mm=-4.0, smoothing=1.5)
    assert preview == pytest.approx(-4.0, abs=0.2)


# ─────────────────────── Undo ───────────────────────

def test_undo_restores_grid_before_lower_adjustment():
    model = _make_model(baseline_mm=-5.0)
    mask = _rect_mask(10, 20, 4, 12)
    before = model.grd.grid.copy()
    model.apply_arch("right", ArchParams(mask=mask, height_mm=-3.0, smoothing=0.5))

    assert model.undo() is True
    assert np.array_equal(model.grd.grid, before)
