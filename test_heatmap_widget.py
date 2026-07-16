"""HeatmapWidgetの選択操作（ペイント式ドラッグ・Ctrl解除・Shift矩形選択）のユニットテスト
（TDD: RED → GREEN）。既存のペイント式選択・Ctrl解除は非デグレ確認、Shift矩形選択は新規機能。"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication

from heatmap_widget import HeatmapWidget


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def hm(qapp):
    w = HeatmapWidget("テスト")
    w.set_select_mode(True)
    return w


def _cell_center(widget: HeatmapWidget, row: int, col: int) -> QPointF:
    x, y = widget._grid_to_px(row, col)
    half = widget._cell_px / 2
    return QPointF(x + half, y + half)


def _mouse_event(widget, event_type, row, col, modifiers=Qt.KeyboardModifier.NoModifier):
    pos = _cell_center(widget, row, col)
    button = Qt.MouseButton.LeftButton
    return QMouseEvent(event_type, pos, pos, button, button, modifiers)


def _press(widget, row, col, modifiers=Qt.KeyboardModifier.NoModifier):
    widget.mousePressEvent(_mouse_event(widget, QEvent.Type.MouseButtonPress, row, col, modifiers))


def _move(widget, row, col, modifiers=Qt.KeyboardModifier.NoModifier):
    widget.mouseMoveEvent(_mouse_event(widget, QEvent.Type.MouseMove, row, col, modifiers))


def _release(widget, row, col, modifiers=Qt.KeyboardModifier.NoModifier):
    widget.mouseReleaseEvent(_mouse_event(widget, QEvent.Type.MouseButtonRelease, row, col, modifiers))


def _drag(widget, path, modifiers=Qt.KeyboardModifier.NoModifier):
    (r0, c0), *rest = path
    _press(widget, r0, c0, modifiers)
    for r, c in rest:
        _move(widget, r, c, modifiers)
    last = rest[-1] if rest else (r0, c0)
    _release(widget, *last, modifiers)


# ─────────────────────── 既存挙動の非デグレ ───────────────────────

def test_normal_drag_paints_only_the_visited_cells(hm):
    _drag(hm, [(5, 5), (5, 6), (5, 7)])

    mask = hm.get_selection_mask()
    assert mask[5, 5] and mask[5, 6] and mask[5, 7]
    assert mask.sum() == 3


def test_ctrl_drag_deselects_previously_selected_cells(hm):
    _drag(hm, [(5, 5), (5, 6), (5, 7)])

    _drag(hm, [(5, 5), (5, 6)], modifiers=Qt.KeyboardModifier.ControlModifier)

    mask = hm.get_selection_mask()
    assert not mask[5, 5]
    assert not mask[5, 6]
    assert mask[5, 7]


# ─────────────────────── Shift矩形選択（新規） ───────────────────────

def test_shift_drag_selects_bounding_rectangle(hm):
    _drag(hm, [(5, 5), (8, 9)], modifiers=Qt.KeyboardModifier.ShiftModifier)

    mask = hm.get_selection_mask()
    expected = np.zeros((32, 16), dtype=bool)
    expected[5:9, 5:10] = True
    assert np.array_equal(mask, expected)


def test_shift_ctrl_drag_deselects_rectangle(hm):
    full = np.zeros((32, 16), dtype=bool)
    full[0:15, 0:15] = True
    hm._sel_mask = full.copy()

    modifiers = Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier
    _drag(hm, [(5, 5), (8, 9)], modifiers=modifiers)

    mask = hm.get_selection_mask()
    assert not mask[5:9, 5:10].any()
    assert mask[0, 0]  # 矩形範囲外は選択状態を維持


def test_rect_drag_does_not_mutate_mask_until_release(hm):
    _press(hm, 5, 5, modifiers=Qt.KeyboardModifier.ShiftModifier)
    _move(hm, 8, 9, modifiers=Qt.KeyboardModifier.ShiftModifier)

    assert not hm.get_selection_mask().any()


def test_state_resets_after_rect_drag_so_next_plain_drag_paints_normally(hm):
    _drag(hm, [(5, 5), (8, 9)], modifiers=Qt.KeyboardModifier.ShiftModifier)

    assert hm._is_rect_selecting is False
    assert hm._rect_start is None

    _drag(hm, [(20, 3)])

    mask = hm.get_selection_mask()
    assert mask[20, 3]
    assert mask.sum() == (5 * 4) + 1  # 矩形4x5 + 通常ドラッグ1セル


# ─────────────────────── set_reference_mask（矯正範囲の参考表示・新規） ───────────────────────

def test_set_reference_mask_stores_mask(hm):
    mask = np.zeros((32, 16), dtype=bool)
    mask[3, 4] = True

    hm.set_reference_mask(mask)

    assert hm._reference_mask[3, 4]


def test_set_reference_mask_none_clears_it(hm):
    mask = np.zeros((32, 16), dtype=bool)
    mask[3, 4] = True
    hm.set_reference_mask(mask)

    hm.set_reference_mask(None)

    assert hm._reference_mask is None


def test_set_reference_mask_does_not_mutate_sel_mask(hm):
    _drag(hm, [(5, 5)])
    ref = np.zeros((32, 16), dtype=bool)
    ref[3, 4] = True

    hm.set_reference_mask(ref)

    mask = hm.get_selection_mask()
    assert mask[5, 5]
    assert not mask[3, 4]
