"""保存完了などを知らせる、自動で消えるトースト通知ウィジェット"""
from PyQt6.QtWidgets import QWidget, QLabel, QHBoxLayout
from PyQt6.QtCore import QTimer, Qt

_DEFAULT_DURATION_MS = 2500
_BOTTOM_MARGIN = 16


class ToastWidget(QWidget):
    """親ウィジェットの子として表示する、フォーカスを奪わない通知。"""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        self._label = QLabel(self)
        self._label.setStyleSheet(
            "background-color: rgba(30, 30, 30, 230);"
            "color: white;"
            "border-radius: 6px;"
            "padding: 10px 18px;"
            "font-size: 12px;"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

        self.hide()

    def show_message(self, text: str, duration_ms: int = _DEFAULT_DURATION_MS) -> None:
        self._label.setText(text)
        self.adjustSize()
        self._reposition()
        self.show()
        self.raise_()
        self._timer.stop()
        self._timer.start(duration_ms)

    def _reposition(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        x = (parent.width() - self.width()) // 2
        y = parent.height() - self.height() - _BOTTOM_MARGIN
        self.move(max(x, 0), max(y, 0))
