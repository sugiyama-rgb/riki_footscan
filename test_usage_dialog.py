"""使い方メニュー（UsageDialog / format_usage_text）のユニットテスト（TDD: RED → GREEN）"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

from usage_dialog import UsageSection, UsageDialog, format_usage_text, format_usage_html


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


# ─────────────────────── format_usage_html（タイトル強調・新規） ───────────────────────

def test_format_usage_html_wraps_title_in_bold_and_brackets():
    sections = [UsageSection(title="基本の流れ", items=["項目A"])]
    html = format_usage_html(sections)
    assert "<b>【基本の流れ】</b>" in html


def test_format_usage_html_puts_title_and_each_item_in_separate_p_tags():
    sections = [UsageSection(title="基本の流れ", items=["項目A", "項目B"])]
    html = format_usage_html(sections)
    assert html.count("<p>") == 3  # タイトル1 + 項目2


def test_format_usage_html_escapes_special_characters_in_title_and_items():
    sections = [UsageSection(title="A<B>", items=["X&Y"])]
    html = format_usage_html(sections)
    assert "A<B>" not in html
    assert "X&Y" not in html
    assert "A&lt;B&gt;" in html
    assert "X&amp;Y" in html


def test_format_usage_html_empty_list_returns_empty_string():
    assert format_usage_html([]) == ""


def test_usage_dialog_uses_html_and_shows_bracketed_title(qapp):
    dlg = UsageDialog([UsageSection(title="基本の流れ", items=["項目A"])])
    assert "【基本の流れ】" in dlg._text_edit.toPlainText()


def test_usage_dialog_items_stay_on_separate_lines(qapp):
    dlg = UsageDialog([UsageSection(title="基本の流れ", items=["項目A", "項目B"])])
    text = dlg._text_edit.toPlainText()
    lines = [line for line in text.split("\n") if line.strip()]
    assert "・項目A" in lines
    assert "・項目B" in lines
