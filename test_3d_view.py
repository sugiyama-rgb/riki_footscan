"""3D表示の視点定義・サーフェス描画ロジックのユニットテスト（TDD: RED → GREEN）"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure

from main import _lookup_view, _render_foot_surface, _SPLIT_VIEW_LABELS, _initial_azim


def test_initial_azim_left_foot_is_unchanged():
    assert _initial_azim(0) == -55


def test_initial_azim_right_foot_is_mirrored():
    assert _initial_azim(1) == -125


def test_lookup_view_returns_matching_spec():
    # 後方＝かかと側から（以前は前方と定義が逆だった）
    assert _lookup_view("後方") == ("後方", 0, -90, False, False, False)
    assert _lookup_view("前方") == ("前方", 0, 90, False, False, False)
    assert _lookup_view("内側") == ("内側", 0, -145, False, True, False)
    assert _lookup_view("外側") == ("外側", 0, -35, False, False, True)


def test_lookup_view_raises_keyerror_for_unknown_label():
    with pytest.raises(KeyError):
        _lookup_view("存在しない視点")


def test_split_view_labels_are_all_lookupable():
    for label in _SPLIT_VIEW_LABELS:
        _lookup_view(label)  # KeyErrorが出なければOK


@pytest.fixture
def ax():
    fig = Figure()
    return fig.add_subplot(projection='3d')


@pytest.fixture
def grid():
    return np.full((4, 4), 5.0)


@pytest.fixture
def base_grid():
    return np.zeros((4, 4))


def test_render_without_diff_returns_wireframe_and_no_base(ax, grid, base_grid):
    wireframe, base_artist = _render_foot_surface(ax, grid, base_grid, diff_enabled=False)
    assert wireframe is not None
    assert base_artist is None


def test_render_with_diff_returns_wireframe_and_base(ax, grid, base_grid):
    wireframe, base_artist = _render_foot_surface(ax, grid, base_grid, diff_enabled=True)
    assert wireframe is not None
    assert base_artist is not None


def test_render_twice_without_old_artists_does_not_error(ax, grid, base_grid):
    _render_foot_surface(ax, grid, base_grid, diff_enabled=False)
    _render_foot_surface(ax, grid, base_grid, diff_enabled=False)
    # 古いartistを渡さなければ蓄積される（呼び出し側の責務）ことの確認
    assert len(ax.collections) == 2


def test_render_with_old_artists_replaces_instead_of_accumulating(ax, grid, base_grid):
    wf1, bwf1 = _render_foot_surface(ax, grid, base_grid, diff_enabled=True)
    before_count = len(ax.collections)
    wf2, bwf2 = _render_foot_surface(
        ax, grid, base_grid, diff_enabled=True,
        old_wireframe=wf1, old_base_wireframe=bwf1,
    )
    after_count = len(ax.collections)
    assert after_count == before_count
    assert wf2 is not wf1
    assert bwf2 is not bwf1


# ─────────────────────── 実寸表示（既存の「3D表示を開く」） ───────────────────────

@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("deepest_mm", [5.0, 20.0])
def test_3d_dialog_split_axes_are_true_scale_regardless_of_foot_depth(qapp, deepest_mm):
    import grd_io
    from foot_model import FootModel
    from main import MainWindow

    grid = np.zeros((grd_io.GRID_ROWS, grd_io.GRID_COLS))
    grid[5:28, 3:13] = -deepest_mm
    grid[37:60, 3:13] = -deepest_mm
    win = MainWindow()
    win._model = FootModel(grd_io.GrdData(grid=grid, patient=grd_io.PatientInfo(), raw_meta_lines=[]))
    fig = Figure()
    win._build_split_3d_axes(fig)
    for ax in win._3d_axes:
        spans = np.array([abs(np.diff(ax.get_xlim3d())[0]), abs(np.diff(ax.get_ylim3d())[0]),
                          abs(np.diff(ax.get_zlim3d())[0])])
        np.testing.assert_allclose(ax._box_aspect / spans, (ax._box_aspect / spans)[0])
        assert sorted(ax.get_zlim3d()) == pytest.approx([-3.0, 0.0])


# ─────────────────────── 「3D表示を開く」の両足1画面 ───────────────────────

def _win_with_model(qapp):
    import grd_io
    from foot_model import FootModel
    from main import MainWindow
    win = MainWindow()
    grid = np.zeros((grd_io.GRID_ROWS, grd_io.GRID_COLS))
    grid[5:28, 3:13] = -8.0
    grid[37:60, 3:13] = -8.0
    win._model = FootModel(grd_io.GrdData(grid=grid, patient=grd_io.PatientInfo(), raw_meta_lines=[]))
    return win


def test_3d_dialog_shows_both_feet_in_one_view(qapp):
    from foot3d_preview import Feet3DView
    win = _win_with_model(qapp)
    win._show_3d_view()
    try:
        assert isinstance(win._3d_feet_view, Feet3DView)
        assert win._3d_feet_view._state is not None
        left, right = win._3d_feet_view._state[:2]
        np.testing.assert_allclose(left[0], win._model.left_grid)
        np.testing.assert_allclose(right[0], win._model.right_grid)
    finally:
        win._3d_dialog.close()


def test_3d_dialog_passes_diff_bases_when_diff_enabled(qapp):
    win = _win_with_model(qapp)
    win._btn_diff.setChecked(True)
    win._show_3d_view()
    try:
        bases = win._3d_feet_view._state[2]
        assert bases is not None
        np.testing.assert_allclose(bases[0], win._model.locked_left_grid)
    finally:
        win._3d_dialog.close()

