"""3D表示の視点定義・サーフェス描画ロジックのユニットテスト（TDD: RED → GREEN）"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure

from main import _lookup_view, _render_foot_surface, _SPLIT_VIEW_LABELS


def test_lookup_view_returns_matching_spec():
    assert _lookup_view("後方") == ("後方", 0, 90, True, False, False)
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
