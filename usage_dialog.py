"""メニューバー「使い方」に表示する操作ガイド"""
from dataclasses import dataclass

from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton


@dataclass(frozen=True)
class UsageSection:
    title: str
    items: list[str]


USAGE_SECTIONS: list[UsageSection] = [
    UsageSection(
        title="基本の流れ",
        items=[
            "「ファイル」からGRDファイルを開く",
            "左側のヒートマップで足の形状を確認しながら、各タブ・パネルの機能で編集する",
            "編集が終わったら「上書き保存」または「名前をつけて保存」で保存する",
        ],
    ),
    UsageSection(
        title="範囲選択の操作（①手動消去・④スムージング・アーチ調整で共通）",
        items=[
            "ドラッグ：1マスずつ選択",
            "Shift+ドラッグ：範囲をまとめて選択（矩形選択）",
            "Ctrl+ドラッグ：選択を1マスずつ解除",
            "Shift+Ctrl+ドラッグ：範囲をまとめて選択解除",
        ],
    ),
    UsageSection(
        title="前処理タブ",
        items=[
            "位置調整：基準位置（踵・第2中足骨）からのずれを、値は変えずに位置のみ補正する",
            "自動ノイズ除去：足形状から孤立したピンデータを自動で除去する",
            "①手動消去：ヒートマップ上でドラッグして選択した範囲を消去する",
            "②左右対称マスク：正常な足を参照にして反対側の形を対称に整形する",
            "③ミラーコピー：片方の足を左右反転して反対側に上書きコピーする",
            "④スムージング：足形状と異なる出っ張り・へこみ（スパイク）を自動でなめらかに補正する",
        ],
    ),
    UsageSection(
        title="アーチ調整タブ",
        items=[
            "アーチ領域をドラッグして選択し、「持ち上げる」か「へこませる（免荷）」を選んで調整量を指定し適用する",
            "「↩ 戻す」「↪ 進む」で直前の操作を取り消し／やり直しできる",
        ],
    ),
    UsageSection(
        title="メタターサルタブ",
        items=[
            "ヒートマップをクリックして配置位置を指定し、高さ・長さ・幅などを設定して適用する",
        ],
    ),
    UsageSection(
        title="患者情報タブ",
        items=[
            "氏名・靴サイズ・性別・左右足サイズを入力し「患者情報を保存」でGRDファイルに書き込む",
        ],
    ),
    UsageSection(
        title="その他",
        items=[
            "調整レイヤー：適用した編集をレイヤーとして一覧表示。チェックでON/OFF、右クリックで固定/解除できる",
            "差分表示：編集前後の差分をヒートマップ上で確認できる",
            "3D表示を開く：編集結果を3Dで確認できる",
            "元に戻す／進む／全リセット：編集操作を取り消し・やり直し、またはすべて取り消す",
        ],
    ),
]


def format_usage_text(sections: list[UsageSection]) -> str:
    blocks = []
    for section in sections:
        lines = [section.title] + [f"・{item}" for item in section.items]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


class UsageDialog(QDialog):
    def __init__(self, sections: list[UsageSection], parent=None):
        super().__init__(parent)
        self.setWindowTitle("使い方")
        self.setMinimumWidth(480)
        self.setMinimumHeight(560)
        layout = QVBoxLayout(self)

        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setPlainText(format_usage_text(sections))
        layout.addWidget(self._text_edit)

        btn_close = QPushButton("閉じる")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)
