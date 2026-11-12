"""メニューバー「使い方」に表示する操作ガイド"""
import html as html_lib
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
            "対象足：右側パネルの［左足］［右足］ボタンで選ぶ。位置調整・手動消去・スムージング・アーチ調整のすべてに共通で効く。切り替えるとヒートマップ上の選択範囲はクリアされる",
        ],
    ),
    UsageSection(
        title="範囲選択の操作（手動消去・スムージング・アーチ調整で共通）",
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
            "位置調整の「左へ1cm」「右へ1cm」：対象足ボタンで足を選び、押すたびに1cmずつ動く（2回で2cm）。1cm戻すときは反対のボタンを押す（「元に戻す」は位置調整をまとめて取り消す）",
            "ミラーコピー：「左足 → 右足にコピー」「右足 → 左足にコピー」のボタンで、片方の足を左右反転して反対側に上書きコピーする",
            "自動ノイズ除去：足形状から孤立したピンデータを自動で除去する",
            "手動消去：ヒートマップ上でドラッグして選択した範囲を消去する",
            "スムージング：足形状と異なる出っ張り・へこみ（スパイク）を自動でなめらかに補正する",
        ],
    ),
    UsageSection(
        title="アーチ調整タブ",
        items=[
            "アーチ領域をドラッグして選択し、「持ち上げる」か「へこませる（免荷）」を選んで調整量を指定し適用する",
            "「反対足にも適用（ミラー）」は最初からONになっている。片足だけに適用したいときはチェックを外す",
            "ヒートマップ右の3Dで、両足の選択範囲（黄色）と適用前の上がり具合を実寸で確認できる。ドラッグで両足まとめて回転、右ドラッグで拡大・縮小（上へ拡大／下へ縮小）、Shift＋ドラッグで移動、ダブルクリックで標準の角度・大きさ・位置に戻す",
            "「↩ 戻す」「↪ 進む」で直前の操作を取り消し／やり直しできる",
        ],
    ),
    UsageSection(
        title="患者情報",
        items=[
            "メニューのすぐ下に常に表示される。氏名・靴サイズ・性別・左右足サイズを入力すると、入力欄から離れた時点で反映され、ファイルを保存するときに一緒に書き込まれる",
        ],
    ),
    UsageSection(
        title="その他",
        items=[
            "調整レイヤー：適用した編集をレイヤーとして一覧表示。チェックでON/OFF、右クリックで固定/解除できる",
            "差分表示：編集前後の差分をヒートマップ上で確認できる",
            "3D表示を開く：編集結果を大きな画面の3D（両足1画面、右の3Dと同じ操作）で確認できる。内側・外側は画面分割モードで左右別に見る",
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


def format_usage_html(sections: list[UsageSection]) -> str:
    blocks = []
    for section in sections:
        title = html_lib.escape(section.title)
        lines = [f"<p><b>【{title}】</b></p>"]
        for item in section.items:
            lines.append(f"<p>・{html_lib.escape(item)}</p>")
        blocks.append("".join(lines))
    return "".join(blocks)


class UsageDialog(QDialog):
    def __init__(self, sections: list[UsageSection], parent=None):
        super().__init__(parent)
        self.setWindowTitle("使い方")
        self.setMinimumWidth(480)
        self.setMinimumHeight(560)
        layout = QVBoxLayout(self)

        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setHtml(format_usage_html(sections))
        layout.addWidget(self._text_edit)

        btn_close = QPushButton("閉じる")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)
