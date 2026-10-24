"""選択モード（手動消去・スムージング・アーチ調整）の排他と、アーチ調整プレビュー表示のユニットテスト
（TDD: RED → GREEN）"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

import grd_io
from foot_model import FootModel
from main import MainWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(qapp):
    w = MainWindow()
    grid = np.full((grd_io.GRID_ROWS, grd_io.GRID_COLS), -8.0)
    w._model = FootModel(grd_io.GrdData(grid=grid, patient=grd_io.PatientInfo(), raw_meta_lines=[]))
    return w


def _buttons(win):
    return {
        "erase": win._btn_erase_select,
        "smooth": win._btn_smooth_select,
        "arch": win._btn_arch_select,
    }


@pytest.mark.parametrize("first, second", [
    ("erase", "arch"), ("smooth", "arch"),
    ("arch", "erase"), ("smooth", "erase"),
    ("arch", "smooth"), ("erase", "smooth"),
])
def test_turning_on_another_select_mode_keeps_selection_usable(win, first, second):
    # 対象足はどれも既定の左足
    buttons = _buttons(win)
    buttons[first].setChecked(True)
    buttons[second].setChecked(True)

    assert buttons[second].isChecked()
    assert not buttons[first].isChecked()
    assert win._hm_left._select_mode is True
    assert win._hm_right._select_mode is False


def test_select_mode_follows_target_foot_of_the_active_button(win):
    win._arch_foot_combo.setCurrentIndex(1)  # アーチは右足
    win._btn_erase_select.setChecked(True)   # 消去は左足
    win._btn_arch_select.setChecked(True)
    assert win._hm_right._select_mode is True
    assert win._hm_left._select_mode is False


def _select_arch_cells(win):
    win._tabs.setCurrentIndex(1)
    win._btn_arch_select.setChecked(True)
    hm = win._hm_left
    hm._sel_mask[10:15, 5:8] = True
    hm.selectionChanged.emit(hm._sel_mask.copy())
    assert win._arch_preview_label.text() != ""


def test_clear_selection_clears_preview_label(win):
    _select_arch_cells(win)
    win._clear_arch_selection()
    assert win._arch_preview_label.text() == ""


def test_turning_off_arch_select_mode_clears_preview_label(win):
    _select_arch_cells(win)
    win._btn_arch_select.setChecked(False)
    assert win._arch_preview_label.text() == ""


def test_applying_arch_clears_preview_label(win):
    _select_arch_cells(win)
    win._apply_arch()
    assert win._arch_preview_label.text() == ""


def test_preview_label_comes_back_when_returning_to_arch_tab(win):
    _select_arch_cells(win)
    win._tabs.setCurrentIndex(0)
    assert win._arch_preview_label.text() == ""
    win._tabs.setCurrentIndex(1)
    assert win._arch_preview_label.text() != ""
