"""使い方メニュー（UsageDialog / format_usage_text）のユニットテスト（TDD: RED → GREEN）"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

from usage_dialog import UsageSection, UsageDialog, format_usage_text


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_format_usage_text_single_section():
    sections = [UsageSection(title="基本の流れ", items=["項目A", "項目B"])]
    text = format_usage_text(sections)
    assert text == "基本の流れ\n・項目A\n・項目B"


def test_format_usage_text_multiple_sections_separated_by_blank_line():
    sections = [
        UsageSection(title="基本の流れ", items=["項目A"]),
        UsageSection(title="前処理タブ", items=["項目B"]),
    ]
    text = format_usage_text(sections)
    assert text == "基本の流れ\n・項目A\n\n前処理タブ\n・項目B"


def test_format_usage_text_empty_list_returns_empty_string():
    assert format_usage_text([]) == ""


def test_usage_dialog_title(qapp):
    dlg = UsageDialog([UsageSection(title="基本の流れ", items=["項目A"])])
    assert dlg.windowTitle() == "使い方"


def test_usage_dialog_shows_section_text(qapp):
    dlg = UsageDialog([UsageSection(title="基本の流れ", items=["項目A"])])
    assert "項目A" in dlg._text_edit.toPlainText()
