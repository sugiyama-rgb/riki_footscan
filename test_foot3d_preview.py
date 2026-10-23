"""ヒートマップ横の埋め込み3Dプレビュー（Foot3DPreview / compute_highlight / upsample_surface）の
ユニットテスト（TDD: RED → GREEN）"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from foot3d_preview import (
    MIRROR_RGB, SELECT_RGB, Foot3DPreview, apply_true_scale, compute_highlight, depth_brightness,
    upsample_surface,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _mask_at(*cells: tuple[int, int]) -> np.ndarray:
    mask = np.zeros((32, 16), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


# ─────────────────────── compute_highlight（計測点ごとの強調色と重み） ───────────────────────

def test_highlight_shape():
    hl = compute_highlight(_mask_at((5, 5)), np.zeros((32, 16)), mirror=False)
    assert hl.shape == (32, 16, 4)


def test_only_selected_point_is_highlighted_semi_transparently():
    hl = compute_highlight(_mask_at((5, 5)), np.zeros((32, 16)), mirror=False)
    np.testing.assert_allclose(hl[5, 5, :3], SELECT_RGB)
    # 塗りつぶしだと下の陰影・網目が見えないため、選択色は半透明で重ねる
    assert 0.4 < hl[5, 5, 3] < 1.0
    for r, c in [(4, 5), (6, 5), (5, 4), (5, 6), (4, 4)]:
        assert hl[r, c, 3] == pytest.approx(0.0)


def test_mirror_uses_mirror_color():
    hl = compute_highlight(_mask_at((5, 5)), np.zeros((32, 16)), mirror=True)
    np.testing.assert_allclose(hl[5, 5, :3], MIRROR_RGB)


def test_selection_on_last_row_and_column_is_highlighted():
    hl = compute_highlight(_mask_at((31, 15)), np.zeros((32, 16)), mirror=False)
    assert hl[31, 15, 3] == pytest.approx(hl[..., 3].max())


def test_changed_but_unselected_point_is_partially_highlighted():
    delta = np.zeros((32, 16))
    delta[20, 10] = 2.0
    hl = compute_highlight(_mask_at((5, 5)), delta, mirror=False)
    assert 0.0 < hl[20, 10, 3] < hl[5, 5, 3]  # 選択範囲そのものより薄い


def test_smaller_change_is_weaker_than_larger_change():
    delta = np.zeros((32, 16))
    delta[20, 10] = 1.3
    delta[20, 12] = 0.3
    hl = compute_highlight(_mask_at((5, 5)), delta, mirror=False)
    assert hl[20, 12, 3] < hl[20, 10, 3]


def test_change_strength_is_absolute_not_relative_to_largest_change():
    # 1mmしか上がらないときに、5mm上がるときと同じ濃さに見えてはいけない
    small = np.zeros((32, 16))
    small[20, 10] = 1.0
    large = np.zeros((32, 16))
    large[20, 10] = 5.0
    w_small = compute_highlight(None, small, mirror=False)[20, 10, 3]
    w_large = compute_highlight(None, large, mirror=False)[20, 10, 3]
    assert w_small == pytest.approx(w_large / 5)


def test_depth_shading_uses_fixed_reference_depth():
    # 同じ深さは、どの足でも同じ暗さ（足ごとの最深部を基準にしない）
    b = depth_brightness(np.array([0.0, 10.0, 20.0, 30.0]))
    assert b[0] == pytest.approx(1.0)
    assert b[1] > b[2]
    assert b[2] == pytest.approx(b[3])  # 20mm以上は同じ（最も暗い）


def test_negligible_change_is_not_highlighted():
    delta = np.zeros((32, 16))
    delta[20, 10] = 0.01
    hl = compute_highlight(_mask_at((5, 5)), delta, mirror=False)
    assert hl[20, 10, 3] == pytest.approx(0.0)


def test_no_mask_and_no_delta_has_no_highlight():
    hl = compute_highlight(None, None, mirror=False)
    assert np.all(hl[..., 3] == 0.0)


# ─────────────────────── upsample_surface（計測点を通る2倍解像度の面） ───────────────────────

def test_upsample_keeps_measured_values_at_points():
    rng = np.random.default_rng(0)
    grid = -rng.uniform(0, 15, size=(32, 16))
    fine = upsample_surface(grid)
    assert fine.shape == (65, 33)
    np.testing.assert_allclose(fine[1::2, 1::2], grid)  # 計測点の値は平均せずそのまま


def test_upsample_midpoints_lie_on_lines_between_points():
    grid = np.zeros((32, 16))
    grid[5, 5] = -8.0
    fine = upsample_surface(grid)
    # 計測点(5,5)と(5,6)の中間 → 2点の平均（白線は点同士を直線で結ぶので、その線上にある）
    assert fine[11, 12] == pytest.approx(-4.0)
    assert fine[12, 11] == pytest.approx(-4.0)  # (5,5)と(6,5)の中間


# ─────────────────────── Foot3DPreview ───────────────────────

@pytest.fixture
def view(qapp):
    w = Foot3DPreview(initial_azim=-55)
    return w


def _grid(value: float = -8.0) -> np.ndarray:
    return np.full((32, 16), value)


def test_repeated_updates_do_not_accumulate_artists(view):
    for v in (-8.0, -7.0, -6.0):
        view.set_data(_grid(v), None, None, mirror=False)
        view.flush()
    assert len(view.axes.collections) == 2  # 青い面＋白い線


def test_preview_draws_surface_and_wire(view):
    mask = _mask_at((10, 5))
    preview = _grid()
    preview[10, 5] = -5.0
    view.set_data(_grid(), preview, mask, mirror=False)
    view.flush()
    assert len(view.axes.collections) == 2


def test_wire_is_drawn_on_top_of_surface(view):
    view.set_data(_grid(), None, None, mirror=False)
    view.flush()
    assert view.axes.computed_zorder is False
    surface, wire = view.axes.collections
    assert wire.get_zorder() > surface.get_zorder()


def test_same_input_does_not_render_again(view):
    view.set_data(_grid(), None, None, mirror=False)
    view.flush()
    count = view.render_count
    view.set_data(_grid(), None, None, mirror=False)
    view.flush()
    assert view.render_count == count


def test_hidden_widget_defers_rendering_until_flush(view):
    view.set_data(_grid(), None, None, mirror=False)
    assert view.render_count == 0
    view.flush()
    assert view.render_count == 1


def test_user_view_angle_is_kept_across_updates(view):
    view.set_data(_grid(-8.0), None, None, mirror=False)
    view.flush()
    view.axes.view_init(elev=10, azim=30)
    view.set_data(_grid(-6.0), None, None, mirror=False)
    view.flush()
    assert view.axes.elev == pytest.approx(10)
    assert view.axes.azim == pytest.approx(30)


def test_reset_view_restores_initial_angle(view):
    view.axes.view_init(elev=10, azim=30)
    view.reset_view()
    assert view.axes.azim == pytest.approx(-55)


def _assert_true_scale(ax):
    """1cm が縦・横・高さのどの方向でも同じ長さで描かれる（実寸 1:1:1）こと。"""
    spans = np.array([abs(np.diff(ax.get_xlim3d())[0]), abs(np.diff(ax.get_ylim3d())[0]),
                      abs(np.diff(ax.get_zlim3d())[0])])
    np.testing.assert_allclose(ax._box_aspect / spans, (ax._box_aspect / spans)[0])


def test_apply_true_scale_uses_fixed_height_range_and_ortho(qapp):
    from matplotlib.figure import Figure
    ax = Figure().add_subplot(projection="3d")
    apply_true_scale(ax, rows=32, cols=16)
    _assert_true_scale(ax)
    assert ax.get_zlim3d() == pytest.approx((-3.0, 0.0))  # 0〜30mm（最大20mm＋へこませる最大10mm）
    assert ax._focal_length == np.inf  # 平行投影（遠近感で比率を見誤らない）


@pytest.mark.parametrize("deepest_mm", [5.0, 20.0])
def test_embedded_view_is_true_scale_regardless_of_foot_depth(qapp, deepest_mm):
    grid = np.zeros((32, 16))
    grid[5:28, 3:13] = -deepest_mm
    w = Foot3DPreview(initial_azim=-90)
    w.set_data(grid, None, None, mirror=False)
    w.flush()
    _assert_true_scale(w.axes)
    assert w.axes.get_zlim3d() == pytest.approx((-3.0, 0.0))


def test_axis_range_is_kept_when_preview_changes_depth(view):
    view.set_data(_grid(), None, None, mirror=False)
    view.flush()
    before = view.axes.get_zlim3d()
    view.set_data(_grid(), _grid(-2.0), _mask_at((5, 5)), mirror=False)
    view.flush()
    np.testing.assert_allclose(view.axes.get_zlim3d(), before)


# ─────────────────────── 視点切替（後面） ───────────────────────

def test_set_view_with_flip_inverts_x_and_survives_rerender(view):
    view.set_data(_grid(-8.0), None, None, mirror=False)
    view.flush()
    view.set_view(elev=0, azim=90, flip_x=True)
    view.set_data(_grid(-6.0), None, None, mirror=False)
    view.flush()
    x0, x1 = view.axes.get_xlim3d()
    assert x0 > x1
    assert view.axes.elev == pytest.approx(0)
    assert view.axes.azim == pytest.approx(90)


def test_reset_view_clears_flip(view):
    view.set_data(_grid(), None, None, mirror=False)
    view.flush()
    view.set_view(elev=0, azim=90, flip_x=True)
    view.reset_view()
    x0, x1 = view.axes.get_xlim3d()
    assert x0 < x1


def test_select_color_contrasts_with_blue_surface():
    from foot3d_preview import SURFACE_RGB
    # 青い面の上で埋もれないよう、選択色は青成分が小さく赤・緑成分が大きい（黄系）
    assert SELECT_RGB[2] < 0.3 and SELECT_RGB[0] > 0.8 and SELECT_RGB[1] > 0.8
    assert SURFACE_RGB == (0.0, 0.0, 1.0)


def test_wire_is_faded_in_back_view_and_restored_in_top_view(view):
    view.set_data(_grid(), None, None, mirror=False)
    view.flush()
    view.set_view(elev=0, azim=90, flip_x=True)
    assert view.axes.collections[1].get_alpha() < 1.0
    view.reset_view()
    assert view.axes.collections[1].get_alpha() == pytest.approx(1.0)
