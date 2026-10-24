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
        date="2026.10.24",
        items=[
            "ヒートマップの右に、両足を1つの画面で見られる3D表示（Ultra Foot の立体図と同じ見た目）を追加しました。アーチ調整で選択した範囲（黄色）と、適用前の上がり具合（へこみ具合）をその場で確認できます。ドラッグで両足まとめて回転（裏返りません）、右ドラッグで拡大・縮小、Shift＋ドラッグで移動、ダブルクリックで標準の角度・大きさ・位置（かかと側から斜め）に戻ります。「表示」欄の「標準／真上／後面」で向きを切り替えられます",
            "アプリを最大化した状態で起動するようにしました",
            "位置調整に「左へ1cm」「右へ1cm」ボタンを追加しました。押すたびに1cmずつ動き、今のずれ（例: 右へ2cm）が表示されます",
            "「3D表示を開く」も両足を1つの画面で表示し、右の3Dと同じ操作（回転・拡大縮小・移動）にしました。向きボタンは標準・前方・後方・上側・下側です（内側・外側は画面分割モードで左右別に見られます）。差分表示ONのときは左右同じ物差しで色分けします",
            "「3D表示を開く」の「前方」「後方」が逆になっていたのを修正しました（後方＝かかと側から、前方＝つま先側から見る）",
            "手動消去・スムージングの選択モード中にアーチ調整の選択モードをONにすると範囲を選べなくなる不具合と、「選択をクリア」をしてもプレビュー表示が残る不具合を修正しました",
            "3D表示を実寸どおり（縦・横・高さが同じ縮尺）に修正しました。これまでは足ごとに高さの見え方が変わり、浅い足ほど高さが強調されていたため、見た目より上がって見えることがありました。浅い足は以前より平らに見えますが、これが実際の形です",
        ],
    ),
    ChangelogEntry(
        date="2026.07.15",
        items=[
            "調整レイヤーの右クリックメニューに「削除」を追加しました。不要になった補正を完全に取り除けます",
            "調整レイヤーの右クリックメニューに「範囲を表示」を追加しました。過去の補正範囲を黒白の破線で確認できます",
            "アーチ調整・手動消去・スムージングの「対象足」を選択モードON中に切り替えても、選択モードが正しく引き継がれるように修正しました（従来はOFF→ONし直す必要がありました）",
            "「使い方」ダイアログのセクションタイトルを太字＋【】で強調し、見やすくしました",
        ],
    ),
    ChangelogEntry(
        date="2026.07.13",
        items=[
            "前処理タブに「スムージング」機能を追加しました。足の形状と異なる出っ張り・へこみ（スパイク）を検出して自動で滑らかに補正できます",
            "範囲選択にShift+ドラッグの矩形選択を追加しました。広い範囲も少ない操作で選択できます（手動消去・アーチ調整の選択でも使えます）",
            "3D表示を開いた際の右足の初期カメラ角度を調整し、左足と見比べやすくしました",
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
