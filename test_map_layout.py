"""ヒートマップ2枚と両足3Dの配置計算（compute_map_layout）のユニットテスト（TDD: RED → GREEN）"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from heatmap_widget import MARGIN_LEFT, MARGIN_TOP, cell_px_for
import pytest
from PyQt6.QtWidgets import QApplication

from main import MAP_3D_GAP, MAP_3D_MIN_W, MAP_HM_GAP, compute_map_layout


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _grid_width(h: int) -> int:
    return MARGIN_LEFT + 16 * cell_px_for(10_000, h) + 10


def test_wide_area_places_heatmaps_at_grid_width_and_3d_at_right():
    w, h = 1600, 1000
    left, right, view = compute_map_layout(w, h, show_3d=True)
    hm_w = _grid_width(h)
    cp = cell_px_for(10_000, h)
    assert left == (0, 0, hm_w, h)
    assert right == (hm_w + MAP_HM_GAP, 0, hm_w, h)
    x3d = 2 * hm_w + MAP_HM_GAP + MAP_3D_GAP
    assert view == (x3d, MARGIN_TOP, w - x3d, 32 * cp)  # 縦位置はヒートマップの方眼にそろえる


def test_narrow_area_hides_3d_and_heatmaps_share_width():
    h = 1000
    w = 2 * _grid_width(h) + MAP_HM_GAP + MAP_3D_GAP + MAP_3D_MIN_W - 1
    left, right, view = compute_map_layout(w, h, show_3d=True)
    assert view is None
    assert left[2] + MAP_HM_GAP + right[2] == w
    assert right[0] == left[2] + MAP_HM_GAP


def test_3d_disabled_hides_3d_even_when_wide():
    left, right, view = compute_map_layout(1600, 1000, show_3d=False)
    assert view is None
    assert left[2] + MAP_HM_GAP + right[2] == 1600


# ─────────────────────── 起動直後（ファイル読込前）も3Dの場所を表示 ───────────────────────

def _row(qapp):
    from heatmap_widget import HeatmapWidget
    from main import _MapRow
    row = _MapRow(HeatmapWidget("左足"), HeatmapWidget("右足"))
    row.resize(1600, 1000)
    row._relayout()
    return row


def test_placeholder_fills_3d_area_before_file_is_opened(qapp):
    row = _row(qapp)
    _, _, view_rect = compute_map_layout(1600, 1000, show_3d=True)
    assert not row._placeholder.isHidden()
    g = row._placeholder.geometry()
    assert (g.x(), g.y(), g.width(), g.height()) == view_rect
    assert "GRD" in row._placeholder.text()


def test_3d_view_replaces_placeholder_when_enabled(qapp):
    from PyQt6.QtWidgets import QWidget
    row = _row(qapp)
    view = QWidget()
    row.set_3d_widget(view)
    row.set_3d_enabled(True)
    assert row._placeholder.isHidden()
    assert not view.isHidden()

