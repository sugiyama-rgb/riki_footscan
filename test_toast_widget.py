"""ToastWidget（保存完了などの自動で消える通知）のユニットテスト（TDD: RED → GREEN）"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication, QWidget

from toast_widget import ToastWidget


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def parent(qapp):
    w = QWidget()
    w.resize(800, 600)
    w.show()
    return w


def test_toast_is_hidden_initially(parent):
    toast = ToastWidget(parent)
    assert toast.isVisible() is False


def test_show_message_makes_widget_visible(parent):
    toast = ToastWidget(parent)
    toast.show_message("保存されました")
    assert toast.isVisible() is True


def test_show_message_sets_label_text(parent):
    toast = ToastWidget(parent)
    toast.show_message("保存されました")
    assert toast._label.text() == "保存されました"


def test_show_message_starts_hide_timer(parent):
    toast = ToastWidget(parent)
    toast.show_message("保存されました", duration_ms=1000)
    assert toast._timer.isActive() is True


def test_show_message_again_replaces_text_and_restarts_timer(parent):
    toast = ToastWidget(parent)
    toast.show_message("first", duration_ms=5000)
    toast.show_message("second", duration_ms=5000)
    assert toast._label.text() == "second"
    assert toast._timer.isActive() is True


def test_hide_makes_widget_invisible(parent):
    toast = ToastWidget(parent)
    toast.show_message("保存されました")
    toast.hide()
    assert toast.isVisible() is False
