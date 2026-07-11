"""調整レイヤーの「固定」機能のユニットテスト（TDD: RED → GREEN）"""
import numpy as np
import pytest

import grd_io
from foot_model import FootModel, LayerRecord
from main import _layer_list_label


def _make_model(right_point=None) -> FootModel:
    grid = np.zeros((grd_io.GRID_ROWS, grd_io.GRID_COLS), dtype=np.float64)
    if right_point is not None:
        (r, c), v = right_point
        grid[r, c] = v
    grd = grd_io.GrdData(grid=grid, patient=grd_io.PatientInfo(), raw_meta_lines=[])
    return FootModel(grd)


def _mask_at(r: int, c: int) -> np.ndarray:
    mask = np.zeros((32, 16), dtype=bool)
    mask[r, c] = True
    return mask


# ─────────────────────── LayerRecord のデフォルト値 ───────────────────────

def test_layer_record_defaults_to_unlocked():
    layer = LayerRecord(name="x", operation="erase", params={})
    assert layer.locked is False


# ─────────────────────── toggle_layer_lock ───────────────────────

def test_toggle_layer_lock_flips_locked_state():
    model = _make_model(right_point=((10, 5), -8.0))
    model.erase_cells("right", _mask_at(10, 5))

    assert model.layers[0].locked is False
    model.toggle_layer_lock(0)
    assert model.layers[0].locked is True
    model.toggle_layer_lock(0)
    assert model.layers[0].locked is False


def test_toggle_layer_lock_out_of_range_does_nothing():
    model = _make_model()
    model.toggle_layer_lock(0)
    assert model.layers == []


def test_toggle_layer_lock_does_not_change_active_grid():
    model = _make_model(right_point=((10, 5), -8.0))
    model.erase_cells("right", _mask_at(10, 5))
    before = model.grd.grid.copy()

    model.toggle_layer_lock(0)

    assert np.array_equal(model.grd.grid, before)


# ─────────────────────── set_all_enabled と locked の除外 ───────────────────────

def test_set_all_enabled_false_skips_locked_layers():
    model = _make_model(right_point=((10, 5), -8.0))
    model.erase_cells("right", _mask_at(10, 5))
    model.toggle_layer_lock(0)

    model.set_all_enabled(False)

    assert model.layers[0].enabled is True


def test_set_all_enabled_true_skips_locked_layers_that_were_manually_disabled():
    model = _make_model(right_point=((10, 5), -8.0))
    model.erase_cells("right", _mask_at(10, 5))
    model.toggle_layer_lock(0)
    model.toggle_layer(0)  # 固定中でも個別に手動OFF

    model.set_all_enabled(True)

    assert model.layers[0].enabled is False


# ─────────────────────── 固定中でも個別チェックボックス(toggle_layer)は操作可能 ───────────────────────

def test_toggle_layer_still_works_on_locked_layer():
    model = _make_model(right_point=((10, 5), -8.0))
    model.erase_cells("right", _mask_at(10, 5))
    model.toggle_layer_lock(0)

    model.toggle_layer(0)

    assert model.layers[0].enabled is False


# ─────────────────────── locked_left_grid / locked_right_grid ───────────────────────

def test_locked_grid_ignores_unlocked_layer_effect():
    model = _make_model(right_point=((10, 5), -8.0))
    model.erase_cells("right", _mask_at(10, 5))  # unlockedのまま

    assert model.locked_right_grid[10, 5] == pytest.approx(-8.0)


def test_locked_grid_reflects_locked_layer_effect():
    model = _make_model(right_point=((10, 5), -8.0))
    model.erase_cells("right", _mask_at(10, 5))
    model.toggle_layer_lock(0)

    assert model.locked_right_grid[10, 5] == 0.0


def test_locked_grid_matches_base_grid_when_no_layers_are_locked():
    model = _make_model(right_point=((10, 5), -8.0))
    model.erase_cells("right", _mask_at(10, 5))

    assert np.array_equal(model.locked_right_grid, model.base_right_grid)


def test_locked_grid_preserves_order_with_unlocked_layer_interleaved():
    # position_adjust(dx=2cm)は足ブロック全体を平行移動するため、両方の点が(10,5)→(10,7)、
    # (20,8)→(20,10)へ移動した後の座標を基準にlayer1/layer2のerase対象を指定する。
    grid = np.zeros((grd_io.GRID_ROWS, grd_io.GRID_COLS), dtype=np.float64)
    grid[10, 5] = -8.0  # 位置調整で(10,7)へ移動する点
    grid[20, 8] = -3.0  # 位置調整で(20,10)へ移動する点
    grd = grd_io.GrdData(grid=grid, patient=grd_io.PatientInfo(), raw_meta_lines=[])
    model = FootModel(grd)

    model.apply_position_adjust("right", dx_cm=2.0, dy_cm=0.0, angle_deg=0.0)  # layer0: 自動固定
    model.erase_cells("right", _mask_at(20, 10))  # layer1: 固定しない（移動後の(20,10)を消す）
    model.erase_cells("right", _mask_at(10, 7))   # layer2: これから固定する（移動後の(10,7)を消す）
    model.toggle_layer_lock(2)

    locked = model.locked_right_grid
    assert locked[10, 7] == 0.0                   # layer0の効果を土台にlayer2が正しく適用されている
    assert locked[20, 10] == pytest.approx(-3.0)  # layer1(unlocked)はスキップされ消えていない


# ─────────────────────── apply_position_adjust の自動固定・引き継ぎ ───────────────────────

def test_apply_position_adjust_new_layer_is_locked_by_default():
    model = _make_model(right_point=((10, 5), -8.0))
    model.apply_position_adjust("right", dx_cm=1.0, dy_cm=0.0, angle_deg=0.0)
    assert model.layers[0].locked is True


def test_apply_position_adjust_reapply_preserves_unlocked_override():
    model = _make_model(right_point=((10, 5), -8.0))
    model.apply_position_adjust("right", dx_cm=1.0, dy_cm=0.0, angle_deg=0.0)
    model.toggle_layer_lock(0)
    assert model.layers[0].locked is False

    model.apply_position_adjust("right", dx_cm=2.0, dy_cm=0.0, angle_deg=0.0)

    assert len(model.layers) == 1
    assert model.layers[0].locked is False


def test_apply_position_adjust_reapply_keeps_locked_true():
    model = _make_model(right_point=((10, 5), -8.0))
    model.apply_position_adjust("right", dx_cm=1.0, dy_cm=0.0, angle_deg=0.0)
    model.apply_position_adjust("right", dx_cm=2.0, dy_cm=0.0, angle_deg=0.0)
    assert model.layers[0].locked is True


# ─────────────────────── _layer_list_label（レイヤーリスト表示用ラベル） ───────────────────────

def test_layer_list_label_prefixes_lock_icon_when_locked():
    layer = LayerRecord(name="位置調整 右足", operation="position_adjust", params={}, locked=True)
    assert _layer_list_label(layer) == "\U0001F512 位置調整 右足"


def test_layer_list_label_has_no_icon_when_unlocked():
    layer = LayerRecord(name="手動消去 左足", operation="erase", params={}, locked=False)
    assert _layer_list_label(layer) == "手動消去 左足"


# ─────────────────────── Undo/Redo での固定状態の保持 ───────────────────────

def test_undo_redo_preserves_locked_state():
    model = _make_model(right_point=((10, 5), -8.0))
    model.erase_cells("right", _mask_at(10, 5))
    model.toggle_layer_lock(0)

    assert model.undo() is True
    assert model.redo() is True
    assert model.layers[0].locked is True
