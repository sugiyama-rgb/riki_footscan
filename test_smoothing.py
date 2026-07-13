"""スムージング（スパイク検出＋補正）機能のユニットテスト（TDD: RED → GREEN）"""
import numpy as np
import pytest

import grd_io
from foot_model import FootModel, SmoothParams


def _footprint_block(value: float = -5.0, r0: int = 8, r1: int = 24, c0: int = 2, c1: int = 14) -> np.ndarray:
    """32×16 のブロックのうち矩形領域だけを接触セル(value)にし、残りは無接触(0)にする。
    足の輪郭境界（接触セルと無接触セルの境目）を含むテストに使う。"""
    block = np.zeros((32, 16), dtype=np.float64)
    block[r0:r1, c0:c1] = value
    return block


def _make_model_from_right_block(block: np.ndarray) -> FootModel:
    """block を右足ブロックとしてベースグリッドに埋め込んだモデルを作る。
    FootModel構築後にright_grid経由で直接書き換えても_base_gridには反映されず
    （_recompute()の際に_base_gridから再計算されるため）テストが無意味になる。
    そのため必ず構築前にgrid配列へ値を埋め込む。"""
    grid = np.zeros((grd_io.GRID_ROWS, grd_io.GRID_COLS), dtype=np.float64)
    grid[grd_io.RIGHT_ROWS] = block
    grd = grd_io.GrdData(grid=grid, patient=grd_io.PatientInfo(), raw_meta_lines=[])
    return FootModel(grd)


def _rect_mask(r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    mask = np.zeros((32, 16), dtype=bool)
    mask[r0:r1, c0:c1] = True
    return mask


# ─────────────────────── スパイク検出＋補正 ───────────────────────

def test_isolated_spike_in_interior_is_corrected_toward_local_median():
    block = _footprint_block(value=-5.0)
    block[15, 8] = -15.0  # 周辺(-5.0)と大きく異なる孤立スパイク
    model = _make_model_from_right_block(block)
    mask = _rect_mask(8, 24, 2, 14)

    model.apply_smoothing("right", SmoothParams(mask=mask, threshold_mm=2.0, strength=1.0))

    assert model.right_grid[15, 8] == pytest.approx(-5.0, abs=0.5)


def test_boundary_cells_are_never_flagged_as_spikes():
    """足の輪郭境界セルは、周辺に無接触セルが混ざることで局所中央値が大きくずれるが、
    スパイクではないため補正されてはならない（シニアエンジニアレビュー指摘の回帰テスト）"""
    block = _footprint_block(value=-5.0)
    model = _make_model_from_right_block(block)
    before = model.grd.grid.copy()
    mask = _rect_mask(8, 24, 2, 14)  # 足形状全体（輪郭境界を含む）を選択

    model.apply_smoothing("right", SmoothParams(mask=mask, threshold_mm=2.0, strength=1.0))

    assert np.array_equal(model.grd.grid, before)


def test_gentle_slope_within_threshold_is_not_modified():
    block = _footprint_block(value=-5.0)
    # 緩やかな勾配（アーチ等の正常形状を想定、隣接セル差は約0.1mm）
    for row in range(8, 24):
        block[row, 2:14] = -5.0 - 0.1 * (row - 8)
    model = _make_model_from_right_block(block)
    before = model.right_grid.copy()
    mask = _rect_mask(10, 22, 4, 12)  # 輪郭境界を避けた内側のみ選択

    model.apply_smoothing("right", SmoothParams(mask=mask, threshold_mm=2.0, strength=1.0))

    assert np.array_equal(model.right_grid, before)


def test_spike_outside_mask_is_untouched():
    block = _footprint_block(value=-5.0)
    block[15, 8] = -15.0
    model = _make_model_from_right_block(block)
    mask = _rect_mask(8, 24, 2, 8)  # スパイク位置(col=8)を含まない範囲

    model.apply_smoothing("right", SmoothParams(mask=mask, threshold_mm=2.0, strength=1.0))

    assert model.right_grid[15, 8] == pytest.approx(-15.0, abs=1e-9)


def test_strength_zero_leaves_grid_unchanged():
    block = _footprint_block(value=-5.0)
    block[15, 8] = -15.0
    model = _make_model_from_right_block(block)
    before = model.right_grid.copy()
    mask = _rect_mask(8, 24, 2, 14)

    model.apply_smoothing("right", SmoothParams(mask=mask, threshold_mm=2.0, strength=0.0))

    assert np.array_equal(model.right_grid, before)


def test_strength_half_blends_halfway_to_median():
    block = _footprint_block(value=-5.0)
    block[15, 8] = -15.0
    model = _make_model_from_right_block(block)
    mask = _rect_mask(8, 24, 2, 14)

    model.apply_smoothing("right", SmoothParams(mask=mask, threshold_mm=2.0, strength=0.5))

    # median≈-5.0, 元の値-15.0の中間 = -10.0 付近
    assert model.right_grid[15, 8] == pytest.approx(-10.0, abs=0.5)


# ─────────────────────── mask未指定 ───────────────────────

def test_mask_none_returns_none_and_adds_no_layer():
    model = _make_model_from_right_block(_footprint_block(value=-5.0))
    before = model.grd.grid.copy()

    result = model.apply_smoothing("right", SmoothParams(mask=None))

    assert result is None
    assert np.array_equal(model.grd.grid, before)
    assert len(model.layers) == 0


# ─────────────────────── レイヤーの積み重ね ───────────────────────

def test_repeated_apply_adds_stacked_layers_not_replaced():
    model = _make_model_from_right_block(_footprint_block(value=-5.0))
    mask = _rect_mask(8, 24, 2, 14)

    model.apply_smoothing("right", SmoothParams(mask=mask, threshold_mm=2.0, strength=1.0))
    model.apply_smoothing("right", SmoothParams(mask=mask, threshold_mm=2.0, strength=1.0))

    assert len(model.layers) == 2


# ─────────────────────── 統計 ───────────────────────

def test_apply_smoothing_stats_report_affected_count():
    block = _footprint_block(value=-5.0)
    block[15, 8] = -15.0
    model = _make_model_from_right_block(block)
    mask = _rect_mask(8, 24, 2, 14)

    stats = model.apply_smoothing("right", SmoothParams(mask=mask, threshold_mm=2.0, strength=1.0))

    assert stats["affected"] > 0


def test_apply_smoothing_stats_report_zero_when_no_spike_found():
    model = _make_model_from_right_block(_footprint_block(value=-5.0))
    mask = _rect_mask(8, 24, 2, 14)

    stats = model.apply_smoothing("right", SmoothParams(mask=mask, threshold_mm=2.0, strength=1.0))

    assert stats["affected"] == 0


# ─────────────────────── Undo ───────────────────────

def test_undo_restores_grid_before_smoothing():
    block = _footprint_block(value=-5.0)
    block[15, 8] = -15.0
    model = _make_model_from_right_block(block)
    before = model.grd.grid.copy()
    mask = _rect_mask(8, 24, 2, 14)

    model.apply_smoothing("right", SmoothParams(mask=mask, threshold_mm=2.0, strength=1.0))
    assert model.undo() is True

    assert np.array_equal(model.grd.grid, before)
