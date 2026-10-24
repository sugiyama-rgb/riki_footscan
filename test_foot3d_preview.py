"""両足を1画面に表示する埋め込み3D（Feet3DView / build_scene / rotate_view / compute_highlight /
upsample_surface）のユニットテスト（TDD: RED → GREEN）"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from foot3d_preview import (
    FEET_GAP_CM, INITIAL_VIEW, MIRROR_RGB, SELECT_RGB, VIEW_BACK, VIEW_BOTTOM, VIEW_FRONT, VIEW_TOP,
    Feet3DView,
    apply_true_scale, build_scene, compute_highlight, depth_brightness, rotate_view, upsample_surface,
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


# ─────────────────────── rotate_view（ドラッグ回転・裏返らない） ───────────────────────

@pytest.mark.parametrize("dy_ratio", [-5.0, -0.3, 0.3, 5.0])
def test_rotate_view_keeps_elevation_between_0_and_90(dy_ratio):
    for elev in (0.0, 30.0, 90.0):
        new_elev, _ = rotate_view(elev, -90.0, 0.0, dy_ratio)
        assert 0.0 <= new_elev <= 90.0


def test_dragging_down_tilts_toward_viewer():
    elev, _ = rotate_view(60.0, -90.0, 0.0, -0.1)  # 下へドラッグ（matplotlibのyは上が正）
    assert elev < 60.0


def test_horizontal_drag_changes_only_azimuth():
    elev, azim = rotate_view(30.0, -90.0, 0.1, 0.0)
    assert elev == pytest.approx(30.0)
    assert azim != pytest.approx(-90.0)


# ─────────────────────── build_scene（両足を1つの面に） ───────────────────────

def _grid(value: float = -8.0) -> np.ndarray:
    return np.full((32, 16), value)


def _foot(grid=None, preview=None, mask=None, mirror=False):
    return (_grid() if grid is None else grid, preview, mask, mirror)


PER_FOOT = 2 * 16 + 1  # 片足の細かい面の頂点列数


def _face_col(foot_idx: int, col: int) -> int:
    """計測点の列 col を囲む細かい面の列（右足は区切り列の後ろ）。"""
    return (0 if foot_idx == 0 else PER_FOOT + 1) + 2 * col + 1


def _face_row(row: int) -> int:
    return 2 * (31 - row) + 1  # flipud 後（かかと側が Y=0）


def test_scene_separates_feet_with_nan_column_and_offsets_right_foot():
    scene = build_scene(_foot(), _foot())
    assert scene["z"].shape == (65, 2 * PER_FOOT + 1)
    assert np.all(np.isnan(scene["z"][:, PER_FOOT]))   # 区切り列
    assert not np.isnan(scene["z"][:, :PER_FOOT]).any()
    assert scene["x"][0, PER_FOOT + 1] == pytest.approx(16 + FEET_GAP_CM - 0.5)  # 右足の左端
    assert len(scene["wires"]) == 2


def test_scene_colors_left_selection_on_left_and_mirror_on_right():
    mask = _mask_at((10, 3))
    scene = build_scene(
        _foot(preview=_grid(), mask=mask),
        _foot(preview=_grid(), mask=np.fliplr(mask), mirror=True),
    )
    c = scene["colors"]
    r = _face_row(10)
    yellow = c[r, _face_col(0, 3), :3]
    orange = c[r, _face_col(1, 12), :3]
    plain = c[r, _face_col(0, 12), :3]
    assert yellow[0] > 0.5 and yellow[1] > 0.5 and yellow[2] < 0.5   # 左足の列3に黄色
    assert orange[0] > 0.5 and orange[1] < orange[0] and orange[2] < 0.5  # 右足の列12にオレンジ
    assert plain[0] < 0.2 and plain[2] > plain[0]                    # 左足の列12は青のまま


# ─────────────────────── Feet3DView ───────────────────────

@pytest.fixture
def view(qapp):
    return Feet3DView()


def _set(view, left=None, right=None):
    view.set_data(left or _foot(), right or _foot())


def test_draws_one_surface_and_two_wires(view):
    _set(view)
    view.flush()
    assert len(view.axes.collections) == 3


def test_repeated_updates_do_not_accumulate_artists(view):
    for v in (-8.0, -7.0, -6.0):
        _set(view, _foot(_grid(v)), _foot(_grid(v)))
        view.flush()
    assert len(view.axes.collections) == 3


def test_wires_are_drawn_on_top_of_surface(view):
    _set(view)
    view.flush()
    assert view.axes.computed_zorder is False
    surface, *wires = view.axes.collections
    assert all(w.get_zorder() > surface.get_zorder() for w in wires)


def test_same_input_does_not_render_again(view):
    _set(view)
    view.flush()
    count = view.render_count
    _set(view)
    view.flush()
    assert view.render_count == count


def test_hidden_widget_defers_rendering_until_flush(view):
    _set(view)
    assert view.render_count == 0
    view.flush()
    assert view.render_count == 1


def test_initial_view_is_from_heel_side_at_an_angle(view):
    assert (view.axes.elev, view.axes.azim) == pytest.approx(INITIAL_VIEW)
    assert 0 < INITIAL_VIEW[0] < 90 and INITIAL_VIEW[1] == pytest.approx(-90)


def test_user_view_angle_is_kept_across_updates(view):
    _set(view, _foot(_grid(-8.0)))
    view.flush()
    view.set_view(10, -60)
    _set(view, _foot(_grid(-6.0)))
    view.flush()
    assert (view.axes.elev, view.axes.azim) == pytest.approx((10, -60))


def test_reset_view_restores_initial_angle(view):
    view.set_view(*VIEW_TOP)
    view.reset_view()
    assert (view.axes.elev, view.axes.azim) == pytest.approx(INITIAL_VIEW)


def test_back_view_is_from_heel_side_without_mirroring(view):
    _set(view)
    view.flush()
    view.set_view(*VIEW_BACK)
    assert VIEW_BACK == (0, -90)
    x0, x1 = view.axes.get_xlim3d()
    assert x0 < x1  # 左右反転しない（左足が左に見える）


def test_wire_fades_only_near_side_views(view):
    _set(view)
    view.flush()
    wires = view.axes.collections[1:]
    assert all(w.get_alpha() == pytest.approx(1.0) for w in wires)  # 初期視点では薄くしない
    view.set_view(*VIEW_BACK)
    assert all(w.get_alpha() < 1.0 for w in wires)
    view.set_view(*VIEW_TOP)
    assert all(w.get_alpha() == pytest.approx(1.0) for w in wires)


def test_mouse_drag_rotates_without_flipping(view):
    from types import SimpleNamespace
    _set(view)
    view.flush()
    h = view._canvas.height() or 100
    view._on_mouse_press(SimpleNamespace(button=1, dblclick=False, x=50, y=50))
    view._on_mouse_move(SimpleNamespace(x=50, y=50 - 10 * h))  # 大きく下へドラッグ
    assert view.axes.elev == pytest.approx(0.0)
    view._on_mouse_move(SimpleNamespace(x=50, y=50 + 10 * h))  # 大きく上へドラッグ
    assert view.axes.elev == pytest.approx(90.0)
    view._on_mouse_release(SimpleNamespace(button=1))
    assert view.axes.roll == pytest.approx(0.0)


def test_double_click_resets_view(view):
    from types import SimpleNamespace
    view.set_view(*VIEW_TOP)
    view._on_mouse_press(SimpleNamespace(button=1, dblclick=True, x=0, y=0))
    assert (view.axes.elev, view.axes.azim) == pytest.approx(INITIAL_VIEW)


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
def test_both_feet_view_is_true_scale_regardless_of_foot_depth(view, deepest_mm):
    grid = np.zeros((32, 16))
    grid[5:28, 3:13] = -deepest_mm
    _set(view, _foot(grid), _foot(grid))
    view.flush()
    _assert_true_scale(view.axes)
    assert view.axes.get_zlim3d() == pytest.approx((-3.0, 0.0))
    assert abs(np.diff(view.axes.get_xlim3d())[0]) == pytest.approx(2 * 16 + FEET_GAP_CM)


def test_axis_range_is_kept_when_preview_changes_depth(view):
    _set(view)
    view.flush()
    before = view.axes.get_zlim3d()
    _set(view, _foot(preview=_grid(-2.0), mask=_mask_at((5, 5))))
    view.flush()
    np.testing.assert_allclose(view.axes.get_zlim3d(), before)


def test_select_color_contrasts_with_blue_surface():
    from foot3d_preview import SURFACE_RGB
    # 青い面の上で埋もれないよう、選択色は青成分が小さく赤・緑成分が大きい（黄系）
    assert SELECT_RGB[2] < 0.3 and SELECT_RGB[0] > 0.8 and SELECT_RGB[1] > 0.8
    assert SURFACE_RGB == (0.0, 0.0, 1.0)


# ─────────────────────── 右ドラッグでズーム ───────────────────────

def _press(view, button, y, dblclick=False):
    from types import SimpleNamespace
    view._on_mouse_press(SimpleNamespace(button=button, dblclick=dblclick, x=50, y=y))


def _move(view, y):
    from types import SimpleNamespace
    view._on_mouse_move(SimpleNamespace(x=50, y=y))


def _release(view, button):
    from types import SimpleNamespace
    view._on_mouse_release(SimpleNamespace(button=button))


def _zoom_of(view) -> float:
    return view.zoom


def test_right_drag_up_zooms_in_and_keeps_true_scale(view):
    _set(view)
    view.flush()
    h = view._canvas.height() or 100
    angle = (view.axes.elev, view.axes.azim)
    _press(view, 3, 50)
    _move(view, 50 + h * 0.3)  # 上へ（matplotlibのyは上が正）
    _release(view, 3)
    assert _zoom_of(view) > 1.0
    assert (view.axes.elev, view.axes.azim) == pytest.approx(angle)  # 右ドラッグでは回転しない
    _assert_true_scale(view.axes)


def test_right_drag_down_zooms_out_with_limit(view):
    _set(view)
    view.flush()
    h = view._canvas.height() or 100
    _press(view, 3, 50)
    _move(view, 50 - h * 100)  # 大きく下へ
    _release(view, 3)
    assert 0.0 < _zoom_of(view) < 1.0
    assert _zoom_of(view) >= 0.4  # 縮小しすぎない


def test_zoom_is_kept_across_updates_and_view_buttons(view):
    _set(view, _foot(_grid(-8.0)))
    view.flush()
    h = view._canvas.height() or 100
    _press(view, 3, 50)
    _move(view, 50 + h * 0.3)
    _release(view, 3)
    zoomed = _zoom_of(view)
    _set(view, _foot(_grid(-6.0)))
    view.flush()
    view.set_view(*VIEW_BACK)
    assert _zoom_of(view) == pytest.approx(zoomed)


def test_double_click_resets_zoom(view):
    _set(view)
    view.flush()
    h = view._canvas.height() or 100
    _press(view, 3, 50)
    _move(view, 50 + h * 0.3)
    _release(view, 3)
    _press(view, 1, 50, dblclick=True)
    assert _zoom_of(view) == pytest.approx(1.0)


# ─────────────────────── Shift＋左ドラッグで平行移動 ───────────────────────

def _view_center(view):
    view.axes.apply_aspect()
    xs, ys = view.axes.viewLim.intervalx, view.axes.viewLim.intervaly
    return (xs[0] + xs[1]) / 2, (ys[0] + ys[1]) / 2


def test_shift_drag_pans_without_rotating(view):
    from types import SimpleNamespace
    _set(view)
    view.flush()
    angle = (view.axes.elev, view.axes.azim)
    before = _view_center(view)
    view._on_mouse_press(SimpleNamespace(button=1, dblclick=False, x=50, y=50, key=None, modifiers=frozenset({"shift"})))
    view._on_mouse_move(SimpleNamespace(x=90, y=50))
    view._on_mouse_release(SimpleNamespace(button=1))
    after = _view_center(view)
    assert after[0] < before[0]  # 右へドラッグ → 表示範囲が左へずれる＝絵が右へ動く
    assert after[1] == pytest.approx(before[1])
    assert (view.axes.elev, view.axes.azim) == pytest.approx(angle)


def test_double_click_resets_pan(view):
    from types import SimpleNamespace
    _set(view)
    view.flush()
    before = _view_center(view)
    view._on_mouse_press(SimpleNamespace(button=1, dblclick=False, x=50, y=50, key=None, modifiers=frozenset({"shift"})))
    view._on_mouse_move(SimpleNamespace(x=90, y=80))
    view._on_mouse_release(SimpleNamespace(button=1))
    view._on_mouse_press(SimpleNamespace(button=1, dblclick=True, x=0, y=0, key=None))
    assert _view_center(view) == pytest.approx(before)


# ─────────────────────── 下側からのドラッグ・向きのプリセット ───────────────────────

def test_drag_from_bottom_view_moves_up_continuously():
    elev, _ = rotate_view(VIEW_BOTTOM[0], VIEW_BOTTOM[1], 0.0, 0.05)  # 少し上へ
    assert VIEW_BOTTOM[0] < elev < 0
    elev2, _ = rotate_view(VIEW_BOTTOM[0], VIEW_BOTTOM[1], 0.0, -0.5)  # 下へは下側より下に行かない
    assert elev2 == pytest.approx(VIEW_BOTTOM[0])


def test_view_presets():
    assert VIEW_FRONT == (0, 90)   # つま先側から
    assert VIEW_BOTTOM[0] < 0      # 下から


# ─────────────────────── 差分表示（補正前との差を色分け） ───────────────────────

def test_diff_colors_use_shared_scale_between_feet():
    base = _grid(-8.0)
    left = _grid(-8.0)
    left[10, 5] = -3.0   # 5mm 持ち上げ
    right = _grid(-8.0)
    right[10, 5] = -7.0  # 1mm 持ち上げ
    scene = build_scene(_foot(left), _foot(right), bases=(base, base))
    c = scene["colors"]
    r = _face_row(10)
    big, small, flat = c[r, _face_col(0, 5), :3], c[r, _face_col(1, 5), :3], c[r, _face_col(0, 12), :3]
    assert big.sum() > small.sum() > flat.sum() - 1e-9  # 左右で同じ物差し（5mm の方が明るい）
    assert not np.allclose(big, small)


def test_view_set_data_with_bases_renders(view):
    base = _grid(-8.0)
    raised = _grid(-8.0)
    raised[10, 5] = -3.0
    view.set_data(_foot(raised), _foot(), bases=(base, base))
    view.flush()
    assert len(view.axes.collections) == 3



def test_real_qt_shift_drag_pans_the_view(qapp):
    # 実際の Qt のマウスイベント（Shift付き）を3Dの画面に送り、平行移動することを確かめる
    from PyQt6.QtCore import QEvent, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent
    v = Feet3DView()
    v.resize(400, 800)
    v.show()
    qapp.processEvents()
    _set(v)
    v.flush()
    angle = (v.axes.elev, v.axes.azim)

    def send(kind, x, buttons):
        e = QMouseEvent(kind, QPointF(x, 400), QPointF(x, 400), Qt.MouseButton.LeftButton, buttons,
                        Qt.KeyboardModifier.ShiftModifier)
        QApplication.sendEvent(v._canvas, e)
        qapp.processEvents()

    send(QEvent.Type.MouseButtonPress, 200, Qt.MouseButton.LeftButton)
    send(QEvent.Type.MouseMove, 260, Qt.MouseButton.LeftButton)
    send(QEvent.Type.MouseButtonRelease, 260, Qt.MouseButton.NoButton)
    assert v.axes.pan[0] > 0
    assert (v.axes.elev, v.axes.azim) == pytest.approx(angle)
