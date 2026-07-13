"""メニューバー「更新内容」に表示する更新履歴"""
from dataclasses import dataclass

from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton


@dataclass(frozen=True)
class ChangelogEntry:
    date: str
    items: list[str]


# 新しいエントリはこのリストの先頭に追加すること。
# 従業員向けの表示のため、実装の詳細ではなく「何が変わったか」を平易な日本語で1〜2行に書く。
CHANGELOG_ENTRIES: list[ChangelogEntry] = [
    ChangelogEntry(
        date="2026.07.13",
        items=[
            "前処理タブに「スムージング」機能を追加しました。足の形状と異なる出っ張り・へこみ（スパイク）を検出して自動で滑らかに補正できます",
            "範囲選択にShift+ドラッグの矩形選択を追加しました。広い範囲も少ない操作で選択できます（手動消去・アーチ調整の選択でも使えます）",
        ],
    ),
    ChangelogEntry(
        date="2026.07.11",
        items=[
            "起動時にタイトルバーへバージョン番号を表示するようにしました",
            "アプリアイコンが表示されない不具合を修正しました",
        ],
    ),
]


def format_changelog_text(entries: list[ChangelogEntry]) -> str:
    blocks = []
    for entry in entries:
        lines = [entry.date] + [f"・{item}" for item in entry.items]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


class ChangelogDialog(QDialog):
    def __init__(self, entries: list[ChangelogEntry], parent=None):
        super().__init__(parent)
        self.setWindowTitle("更新内容")
        self.setMinimumWidth(340)
        layout = QVBoxLayout(self)

        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setPlainText(format_changelog_text(entries))
        layout.addWidget(self._text_edit)

        btn_close = QPushButton("閉じる")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)
