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


def test_select_mode_follows_shared_target_foot(win):
    # 対象足は全項目共通。選択モードON中に右足へ切り替えると、選択できる足も右に移る
    win._btn_erase_select.setChecked(True)
    assert win._hm_left._select_mode is True
    win._set_target_foot("right")
    assert win._hm_right._select_mode is True
    assert win._hm_left._select_mode is False
    win._btn_arch_select.setChecked(True)
    assert win._hm_right._select_mode is True
    assert win._hm_left._select_mode is False


def test_switching_target_foot_clears_selection_mirror_and_preview(win):
    _select_arch_cells(win)
    assert win._hm_right._mirror_mask is not None  # ミラー初期ONなので反対足に表示されている
    win._set_target_foot("right")
    assert not win._hm_left.get_selection_mask().any()
    assert win._hm_left._mirror_mask is None
    assert win._hm_right._mirror_mask is None
    assert win._arch_preview_label.text() == ""


def test_angle_guide_not_shown_on_arch_tab_when_switching_foot(win):
    win._model.apply_position_adjust("right", dx_cm=0.0, dy_cm=0.0, angle_deg=3.0)
    win._tabs.setCurrentIndex(1)
    win._set_target_foot("right")
    assert win._hm_left._angle_guide is None
    assert win._hm_right._angle_guide is None
    win._tabs.setCurrentIndex(0)  # 前処理に戻ると表示される
    assert win._hm_right._angle_guide is not None


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
