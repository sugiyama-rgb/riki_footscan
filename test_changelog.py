"""更新内容メニュー（ChangelogDialog / format_changelog_text）のユニットテスト（TDD: RED → GREEN）"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

from changelog import ChangelogEntry, ChangelogDialog, format_changelog_text


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_format_changelog_text_single_entry():
    entries = [ChangelogEntry(date="2026.07.11", items=["項目A", "項目B"])]
    text = format_changelog_text(entries)
    assert text == "2026.07.11\n・項目A\n・項目B"


def test_format_changelog_text_multiple_entries_separated_by_blank_line():
    entries = [
        ChangelogEntry(date="2026.07.11", items=["項目A"]),
        ChangelogEntry(date="2026.07.08", items=["項目B"]),
    ]
    text = format_changelog_text(entries)
    assert text == "2026.07.11\n・項目A\n\n2026.07.08\n・項目B"


def test_format_changelog_text_empty_list_returns_empty_string():
    assert format_changelog_text([]) == ""


def test_changelog_dialog_title(qapp):
    dlg = ChangelogDialog([ChangelogEntry(date="2026.07.11", items=["項目A"])])
    assert dlg.windowTitle() == "更新内容"


def test_changelog_dialog_shows_entry_text(qapp):
    dlg = ChangelogDialog([ChangelogEntry(date="2026.07.11", items=["項目A"])])
    assert "項目A" in dlg._text_edit.toPlainText()
