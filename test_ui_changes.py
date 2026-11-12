"""2026.11.12 の画面変更（対象足の統一・前処理の並び・メタターサル削除・患者情報バー）のユニットテスト"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication, QGroupBox, QMessageBox, QPushButton

import grd_io
from foot_model import FootModel
from main import MainWindow, SettingsDialog, _DEFAULT_SETTINGS


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _make_grd(meta_lines: list[str], patient: grd_io.PatientInfo) -> grd_io.GrdData:
    grid = np.full((grd_io.GRID_ROWS, grd_io.GRID_COLS), -8.0)
    return grd_io.GrdData(grid=grid, patient=patient, raw_meta_lines=meta_lines)


@pytest.fixture
def win(qapp):
    w = MainWindow()
    grd = _make_grd([], grd_io.PatientInfo())
    w._model = FootModel(grd)
    w._load_patient_fields()
    w._patient_bar.setEnabled(True)
    w._dirty = False
    return w


# ── 画面構成 ──

def test_tabs_are_preprocess_and_arch_only(win):
    assert [win._tabs.tabText(i) for i in range(win._tabs.count())] == ["前処理", "アーチ調整"]


def test_preprocess_sections_order(win):
    noise_tab = win._tabs.widget(0)
    titles = [b.title() for b in noise_tab.findChildren(QGroupBox)]
    assert titles == ["位置調整", "ミラーコピー", "自動ノイズ除去", "手動消去", "スムージング"]


def test_right_3d_view_buttons_removed(win):
    labels = {b.text() for b in win.findChildren(QPushButton)}
    assert not {"標準", "真上", "後面"} & labels


def test_arch_mirror_checked_by_default(win):
    assert win._arch_mirror_check.isChecked()


def test_target_foot_defaults_to_left_and_switches(win):
    assert win._target_foot() == "left"
    win._set_target_foot("right")
    assert win._target_foot() == "right"
    assert not win._btn_foot_left.isChecked()


def test_set_target_foot_rejects_unknown_value(win):
    with pytest.raises(ValueError):
        win._set_target_foot("both")


def test_mirror_copy_buttons_copy_in_given_direction(win, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    grd = _make_grd([], grd_io.PatientInfo())
    grd.grid[grd_io.LEFT_ROWS][5, 2] = -12.0
    win._model = FootModel(grd)
    win._apply_mirror_copy("left")
    assert win._model.right_grid[5, 13] == pytest.approx(-12.0)


def test_patient_bar_disabled_until_file_opened(qapp):
    w = MainWindow()
    assert not w._patient_bar.isEnabled()


# ── 患者情報（自動反映） ──

def test_untouched_patient_fields_do_not_change_meta(qapp):
    """性別が空（画面上は Male 表示）のファイルを、触らずに反映しても書き換わらない"""
    meta = ["_(Taro)"] + [""] * 21
    w = MainWindow()
    w._model = FootModel(_make_grd(list(meta), grd_io.PatientInfo(name="Taro", gender="")))
    w._load_patient_fields()
    assert w._commit_patient_fields() is False
    assert w._model.grd.raw_meta_lines == meta
    assert w._model.grd.patient.gender == ""


def test_editing_a_field_updates_only_that_field_and_marks_dirty(qapp):
    w = MainWindow()
    w._model = FootModel(_make_grd(["_(Taro)"] + [""] * 21, grd_io.PatientInfo(name="Taro", gender="")))
    w._load_patient_fields()
    w._dirty = False
    w._pt_shoe_size.setText("25.5")
    w._pt_shoe_size.editingFinished.emit()
    assert w._model.grd.patient.shoe_size == "25.5"
    assert w._model.grd.raw_meta_lines[11] == "25.5"
    assert w._model.grd.patient.gender == ""  # 触っていない性別はそのまま
    assert w._dirty and w._pt_dirty


def test_changing_gender_combo_applies_immediately(win):
    win._pt_gender.setCurrentIndex(1)
    assert win._model.grd.patient.gender == "Female"
    assert win._dirty


def test_loading_fields_does_not_mark_dirty(win):
    win._model.grd.patient.gender = "Female"
    win._load_patient_fields()
    assert not win._dirty and not win._pt_dirty


def test_close_with_unfinished_input_asks_confirmation(win, monkeypatch):
    asked = []
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: asked.append(1) or QMessageBox.StandardButton.No)
    win._pt_name.setText("Hanako")  # editingFinished が来ないまま閉じる
    win.close()
    assert asked
    assert win._model.grd.patient.name == "Hanako"


def test_reset_keeps_patient_unsaved_flag(win, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    win._pt_name.setText("Hanako")
    win._commit_patient_fields()
    win._reset()  # 補正レイヤーは無いが、患者情報の変更は未保存のまま
    assert win._dirty


# ── 設定画面 ──

def test_settings_dialog_keeps_hidden_keys(qapp):
    defaults = {
        **_DEFAULT_SETTINGS,
        "metatarsal": {**_DEFAULT_SETTINGS["metatarsal"], "height_mm": 7.5},
        "last_open_dir": "C:/scans",
    }
    values = SettingsDialog(defaults).get_values()
    assert values["metatarsal"]["height_mm"] == 7.5
    assert values["last_open_dir"] == "C:/scans"
    assert set(values["arch"]) == {"height_mm", "smoothing"}
