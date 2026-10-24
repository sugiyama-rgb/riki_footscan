"""位置調整（左右・前後オフセット＋任意角度）機能のユニットテスト（TDD: RED → GREEN）"""
import numpy as np
import pytest

import grd_io
from foot_model import FootModel
from main import _compute_angle_guide_line


def _make_model(right_point=None, left_point=None) -> FootModel:
    """右足/左足それぞれの32x16ブロック内の(row, col)に負の値を1点だけ置いたGrdDataからFootModelを作る"""
    grid = np.zeros((grd_io.GRID_ROWS, grd_io.GRID_COLS), dtype=np.float64)
    if right_point is not None:
        (r, c), v = right_point
        grid[grd_io.RIGHT_ROWS][r, c] = v
        grid[r, c] = v  # RIGHT_ROWS is slice(0,32) so absolute row == r
    if left_point is not None:
        (r, c), v = left_point
        grid[32 + r, c] = v
    grd = grd_io.GrdData(grid=grid, patient=grd_io.PatientInfo(), raw_meta_lines=[])
    return FootModel(grd)


# ─────────────────────── シフト（整数cm＝整数セル） ───────────────────────

def test_shift_moves_point_to_expected_cell_and_zero_fills_origin():
    model = _make_model(right_point=((10, 5), -8.0))
    model.apply_position_adjust("right", dx_cm=2.0, dy_cm=3.0, angle_deg=0.0)

    right = model.right_grid
    assert right[13, 7] == pytest.approx(-8.0)
    assert right[10, 5] == 0.0


def test_shift_only_affects_target_foot():
    model = _make_model(right_point=((10, 5), -8.0), left_point=((10, 5), -4.0))
    model.apply_position_adjust("right", dx_cm=2.0, dy_cm=3.0, angle_deg=0.0)

    assert model.left_grid[10, 5] == pytest.approx(-4.0)
    assert model.right_grid[10, 5] == 0.0
    assert model.right_grid[13, 7] == pytest.approx(-8.0)


def test_zero_offset_and_zero_angle_leaves_grid_unchanged():
    model = _make_model(right_point=((10, 5), -8.0))
    before = model.grd.grid.copy()
    model.apply_position_adjust("right", dx_cm=0.0, dy_cm=0.0, angle_deg=0.0)
    assert np.array_equal(model.grd.grid, before)


# ─────────────────────── 回転 ───────────────────────

def test_rotation_never_produces_positive_values():
    model = _make_model(right_point=((20, 8), -5.0))
    model.apply_position_adjust("right", dx_cm=0.0, dy_cm=0.0, angle_deg=10.0)
    assert (model.right_grid <= 0.0).all()


def test_rotation_moves_off_center_point():
    model = _make_model(right_point=((5, 14), -6.0))
    original_value_at_point = model.right_grid[5, 14]
    model.apply_position_adjust("right", dx_cm=0.0, dy_cm=0.0, angle_deg=10.0)
    assert model.right_grid[5, 14] != pytest.approx(original_value_at_point)
    assert model.right_grid.shape == (32, 16)


def test_rotation_pivots_around_heel_not_block_center():
    # 回転軸は踵中心（行31・列7.5）。踵付近の点はほとんど動かず、
    # つま先付近の点は大きく動く（軸から離れているため）はず。
    heel_model = _make_model(right_point=((31, 7), -8.0))
    heel_model.apply_position_adjust("right", dx_cm=0.0, dy_cm=0.0, angle_deg=10.0)
    assert heel_model.right_grid[31, 7] == pytest.approx(-8.0, abs=1.0)

    toe_model = _make_model(right_point=((2, 7), -8.0))
    toe_model.apply_position_adjust("right", dx_cm=0.0, dy_cm=0.0, angle_deg=10.0)
    assert abs(toe_model.right_grid[2, 7]) < 2.0


# ─────────────────────── 多重適用（レイヤーの置き換え、劣化防止） ───────────────────────

def test_reapplying_same_foot_replaces_layer_instead_of_stacking():
    model = _make_model(right_point=((10, 5), -8.0))
    model.apply_position_adjust("right", dx_cm=1.0, dy_cm=0.0, angle_deg=0.0)
    model.apply_position_adjust("right", dx_cm=2.0, dy_cm=3.0, angle_deg=0.0)

    assert len(model.layers) == 1
    assert model.right_grid[13, 7] == pytest.approx(-8.0)


def test_different_feet_get_independent_layers():
    model = _make_model(right_point=((10, 5), -8.0), left_point=((10, 5), -4.0))
    model.apply_position_adjust("right", dx_cm=1.0, dy_cm=0.0, angle_deg=0.0)
    model.apply_position_adjust("left", dx_cm=1.0, dy_cm=0.0, angle_deg=0.0)

    assert len(model.layers) == 2


# ─────────────────────── データ消失の検出 ───────────────────────

def test_lost_cells_is_zero_when_nothing_pushed_out_of_bounds():
    model = _make_model(right_point=((10, 5), -8.0))
    stats = model.apply_position_adjust("right", dx_cm=1.0, dy_cm=1.0, angle_deg=0.0)
    assert stats["lost_cells"] == 0


def test_lost_cells_counts_data_pushed_out_of_grid():
    model = _make_model(right_point=((1, 1), -8.0))
    stats = model.apply_position_adjust("right", dx_cm=-3.0, dy_cm=-3.0, angle_deg=0.0)
    assert stats["lost_cells"] == 1


# ─────────────────────── プレビュー（非破壊） ───────────────────────

def test_preview_does_not_mutate_model():
    model = _make_model(right_point=((1, 1), -8.0))
    before = model.grd.grid.copy()
    stats = model.preview_position_adjust("right", dx_cm=-3.0, dy_cm=-3.0, angle_deg=0.0)

    assert stats["lost_cells"] == 1
    assert np.array_equal(model.grd.grid, before)
    assert model.layers == []


# ─────────────────────── Undo ───────────────────────

def test_undo_restores_grid_before_position_adjust():
    model = _make_model(right_point=((10, 5), -8.0))
    before = model.grd.grid.copy()
    model.apply_position_adjust("right", dx_cm=2.0, dy_cm=3.0, angle_deg=0.0)
    assert model.undo() is True
    assert np.array_equal(model.grd.grid, before)


# ─────────────────────── 角度ガイドライン計算（_compute_angle_guide_line） ───────────────────────

def test_angle_guide_line_zero_angle_is_vertical_reference():
    p_toe, p_heel = _compute_angle_guide_line(0.0)
    assert p_toe == pytest.approx((2.0, 7.5))
    assert p_heel == pytest.approx((29.0, 7.5))


def test_angle_guide_line_toe_end_moves_more_than_heel_end():
    # 回転軸は踵中心（行31・列7.5）に近いため、踵側端点はほぼ動かず、
    # つま先側端点は軸から遠く大きく振れるはず。
    p_toe, p_heel = _compute_angle_guide_line(10.0)
    _, toe_col = p_toe
    _, heel_col = p_heel
    assert abs(toe_col - 7.5) > abs(heel_col - 7.5)


def test_angle_guide_line_negative_angle_tilts_opposite_direction():
    p_toe_pos, _ = _compute_angle_guide_line(10.0)
    p_toe_neg, _ = _compute_angle_guide_line(-10.0)
    _, col_pos = p_toe_pos
    _, col_neg = p_toe_neg
    assert (col_pos - 7.5) == pytest.approx(-(col_neg - 7.5))


# ─────────────────────── 現在の位置調整値（1cmボタン用） ───────────────────────

def test_current_position_adjust_is_zero_without_layer():
    model = _make_model(right_point=((10, 5), -8.0))
    assert model.current_position_adjust("right") == (0.0, 0.0, 0.0)


def test_current_position_adjust_returns_that_foot_values():
    model = _make_model(right_point=((10, 5), -8.0), left_point=((10, 5), -8.0))
    model.apply_position_adjust("right", dx_cm=1.0, dy_cm=-1.0, angle_deg=2.0)
    assert model.current_position_adjust("right") == (1.0, -1.0, 2.0)
    assert model.current_position_adjust("left") == (0.0, 0.0, 0.0)


# ─────────────────────── 「左へ1cm／右へ1cm」ボタン ───────────────────────

@pytest.fixture(scope="module")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(qapp):
    from main import MainWindow
    w = MainWindow()
    w._model = _make_model(right_point=((10, 5), -8.0), left_point=((10, 5), -8.0))
    return w


def _position_layers(model):
    return [l for l in model.layers if l.operation == "position_adjust"]


def test_pressing_right_twice_moves_two_cm_with_single_layer(win):
    win._pos_foot_combo.setCurrentIndex(0)  # 左足
    win._nudge_position(+1.0)
    win._nudge_position(+1.0)
    assert win._model.current_position_adjust("left") == (2.0, 0.0, 0.0)
    assert len(_position_layers(win._model)) == 1
    assert win._model.left_grid[10, 7] == pytest.approx(-8.0)  # 2マス右へ
    assert "右へ2cm" in win._pos_current_label.text()
    assert win._pos_dx.value() == pytest.approx(2.0)


def test_pressing_left_after_right_goes_back(win):
    win._pos_foot_combo.setCurrentIndex(1)  # 右足
    win._nudge_position(+1.0)
    win._nudge_position(-1.0)
    assert win._model.current_position_adjust("right")[0] == pytest.approx(0.0)
    assert "ずれなし" in win._pos_current_label.text()


def test_nudge_keeps_forward_backward_and_angle(win):
    win._pos_foot_combo.setCurrentIndex(1)
    win._model.apply_position_adjust("right", dx_cm=0.0, dy_cm=1.0, angle_deg=2.0)
    win._nudge_position(-1.0)
    assert win._model.current_position_adjust("right") == (-1.0, 1.0, 2.0)


def test_nudge_does_not_exceed_limit(win):
    win._pos_foot_combo.setCurrentIndex(1)
    for _ in range(4):
        win._nudge_position(+1.0)
    assert win._model.current_position_adjust("right")[0] == pytest.approx(3.0)


def test_nudge_asks_before_losing_cells_and_cancel_keeps_grid(win, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    win._model = _make_model(right_point=((10, 15), -8.0))  # 右端の列
    win._pos_foot_combo.setCurrentIndex(1)
    asked = []
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: asked.append(1) or QMessageBox.StandardButton.No)
    win._nudge_position(+1.0)
    assert asked
    assert win._model.current_position_adjust("right") == (0.0, 0.0, 0.0)
