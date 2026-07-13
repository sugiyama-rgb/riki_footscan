"""足底ヒートマップ表示ウィジェット"""
import math
import numpy as np
from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QPainterPath
from PyQt6.QtCore import Qt, QPointF, pyqtSignal, QRect


MARGIN_TOP = 30       # 上マージン（列ラベル）
MARGIN_LEFT = 46      # 左マージン（カラーバー＋踵距離ルーラー）
CBAR_W = 12           # カラーバー幅 (px)
MAX_DEPTH = 20.0      # スケール最大 (mm)
MIN_CELL_PX = 12      # 最小セルサイズ
DEFAULT_CELL_PX = 28  # デフォルトセルサイズ


# UltraFoot準拠カラーストップ: (t, R, G, B)
_STOPS = [
    (0.00,   0,   0,   0),   # 黒
    (0.05,   0,   0, 130),   # 濃紺
    (0.15,   0,  20, 200),   # 青
    (0.30,   0, 170, 200),   # シアン（落ち着き）
    (0.45,  20, 190,  20),   # 緑（落ち着き）
    (0.60, 200, 185,   0),   # 黄（落ち着き）
    (0.75, 215,  75,   0),   # 橙
    (0.85, 195,  10,  10),   # 赤
    (1.00, 185,   0, 185),   # マゼンタ
]


def _depth_to_color(v: float) -> QColor:
    """GRD値(0〜-20mm)をUltraFoot準拠の色に変換。0=黒(無接触)"""
    if v >= 0:
        return QColor(0, 0, 0)
    t = min(abs(v) / MAX_DEPTH, 1.0)
    for i in range(len(_STOPS) - 1):
        t0, r0, g0, b0 = _STOPS[i]
        t1, r1, g1, b1 = _STOPS[i + 1]
        if t <= t1:
            ratio = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
            return QColor(
                int(r0 + ratio * (r1 - r0)),
                int(g0 + ratio * (g1 - g0)),
                int(b0 + ratio * (b1 - b0)),
            )
    return QColor(255, 0, 255)


class HeatmapWidget(QWidget):
    cellClicked = pyqtSignal(int, int)   # (row, col) グリッド座標
    selectionChanged = pyqtSignal(object)  # bool mask (32×16)

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self._grid: np.ndarray = np.zeros((32, 16))
        self._overlay: np.ndarray | None = None
        self._overlay_boundary: np.ndarray | None = None
        self._sel_mask: np.ndarray = np.zeros((32, 16), dtype=bool)
        self._is_selecting = False
        self._is_deselecting = False
        self._select_mode = False
        self._is_rect_selecting = False
        self._rect_start: tuple | None = None
        self._rect_current: tuple | None = None
        self._rect_subtract = False
        self._meta_center: tuple | None = None
        self._mirror_mask: np.ndarray | None = None
        self._angle_guide: tuple | None = None
        self._diff_grid: np.ndarray | None = None
        self._cell_px: int = DEFAULT_CELL_PX

        rows, cols = 32, 16
        self.setMinimumSize(
            MARGIN_LEFT + cols * MIN_CELL_PX + 10,
            rows * MIN_CELL_PX + MARGIN_TOP + 30,
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _compute_cell_px(self) -> int:
        rows, cols = 32, 16
        avail_w = self.width() - MARGIN_LEFT - 10
        avail_h = self.height() - MARGIN_TOP - 30
        if avail_w <= 0 or avail_h <= 0:
            return DEFAULT_CELL_PX
        return max(MIN_CELL_PX, min(avail_w // cols, avail_h // rows, 50))

    def resizeEvent(self, event):
        self._cell_px = self._compute_cell_px()
        self.update()
        super().resizeEvent(event)

    def set_grid(self, grid: np.ndarray):
        self._grid = grid.copy()
        self.update()

    def set_diff_grid(self, diff: np.ndarray | None) -> None:
        self._diff_grid = diff
        self.update()

    def set_overlay(self, overlay: np.ndarray | None, boundary_mask: np.ndarray | None = None):
        self._overlay = overlay
        self._overlay_boundary = boundary_mask
        self.update()

    def set_select_mode(self, enabled: bool):
        self._select_mode = enabled
        if not enabled:
            self._sel_mask[:] = False
        self.update()

    def clear_selection(self):
        self._sel_mask[:] = False
        self.update()

    def get_selection_mask(self) -> np.ndarray:
        return self._sel_mask.copy()

    def set_meta_center(self, rc: tuple | None):
        self._meta_center = rc
        self.update()

    def set_mirror_mask(self, mask: np.ndarray | None):
        self._mirror_mask = mask
        self.update()

    def set_angle_guide(self, line: tuple | None):
        self._angle_guide = line  # ((row1, col1), (row2, col2)) のグリッド座標、Noneで非表示
        self.update()

    # ─── 座標変換 ───
    def _grid_to_px(self, row: int, col: int) -> tuple[int, int]:
        x = MARGIN_LEFT + col * self._cell_px
        y = MARGIN_TOP + row * self._cell_px
        return x, y

    def _px_to_grid(self, px: float, py: float) -> tuple[int, int]:
        col = int((px - MARGIN_LEFT) // self._cell_px)
        row = int((py - MARGIN_TOP) // self._cell_px)
        return row, col

    def _build_overlay_path(self, mask: np.ndarray, rows: int, cols: int, cp: int) -> QPainterPath | None:
        """bool マスクの境界をCatmull-Romスプラインで滑らかに返す"""
        points: list[QPointF] = []

        for r in range(rows - 1):
            for c in range(cols):
                if mask[r, c] != mask[r + 1, c]:
                    x0, y0 = self._grid_to_px(r, c)
                    points.append(QPointF(x0 + cp / 2, y0 + cp - 0.5))

        for r in range(rows):
            for c in range(cols - 1):
                if mask[r, c] != mask[r, c + 1]:
                    x0, y0 = self._grid_to_px(r, c)
                    points.append(QPointF(x0 + cp - 0.5, y0 + cp / 2))

        if len(points) < 3:
            return None

        cx = sum(pt.x() for pt in points) / len(points)
        cy = sum(pt.y() for pt in points) / len(points)
        points.sort(key=lambda pt: math.atan2(pt.y() - cy, pt.x() - cx))

        n = len(points)
        path = QPainterPath()
        path.moveTo(points[0])
        for i in range(n):
            p0 = points[(i - 1) % n]
            p1 = points[i]
            p2 = points[(i + 1) % n]
            p3 = points[(i + 2) % n]
            ctrl1 = QPointF(p1.x() + (p2.x() - p0.x()) / 6, p1.y() + (p2.y() - p0.y()) / 6)
            ctrl2 = QPointF(p2.x() - (p3.x() - p1.x()) / 6, p2.y() - (p3.y() - p1.y()) / 6)
            path.cubicTo(ctrl1, ctrl2, p2)
        path.closeSubpath()
        return path

    # ─── 描画 ───
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        rows, cols = self._grid.shape
        cp = self._cell_px

        # 背景: 黒
        p.fillRect(self.rect(), QColor(0, 0, 0))

        # ── カラーバー（左マージン内） ──
        bar_h = rows * cp
        for py in range(bar_h):
            t = 1.0 - py / bar_h
            color = _depth_to_color(-t * MAX_DEPTH)
            p.fillRect(0, MARGIN_TOP + py, CBAR_W, 1, color)

        # カラーバー深さラベル
        font = QFont()
        font.setPointSize(6)
        p.setFont(font)
        p.setPen(QPen(QColor(180, 180, 180), 1))
        for depth_mm in (0, 5, 10, 15, 20):
            t = depth_mm / MAX_DEPTH
            py = MARGIN_TOP + int(bar_h * (1.0 - t))
            p.drawLine(CBAR_W, py, CBAR_W + 2, py)
            p.drawText(CBAR_W + 3, py + 4, f"{depth_mm}")

        # ── グリッドセル ──
        for r in range(rows):
            for c in range(cols):
                v = self._grid[r, c]
                color = _depth_to_color(v)
                x, y = self._grid_to_px(r, c)
                p.fillRect(x, y, cp - 1, cp - 1, color)
                if self._sel_mask[r, c]:
                    p.fillRect(x, y, cp - 1, cp - 1, QColor(255, 255, 255, 100))
                if self._mirror_mask is not None and self._mirror_mask[r, c]:
                    p.fillRect(x, y, cp - 1, cp - 1, QColor(255, 180, 0, 80))

        # 差分オーバーレイ（元データとの差分を半透明カラーで表示）
        if self._diff_grid is not None:
            _DIFF_SCALE = 5.0
            for r in range(rows):
                for c in range(cols):
                    delta = float(self._diff_grid[r, c])
                    if abs(delta) < 0.1:
                        continue
                    intensity = min(abs(delta) / _DIFF_SCALE, 1.0)
                    alpha = int(intensity * 180)
                    x, y = self._grid_to_px(r, c)
                    color = QColor(0, 204, 102, alpha) if delta < 0 else QColor(255, 102, 0, alpha)
                    p.fillRect(x, y, cp - 1, cp - 1, color)

        # オーバーレイ（メタターサルプレビュー）- 滑らかな輪郭
        if self._overlay is not None:
            boundary = self._overlay_boundary if self._overlay_boundary is not None else (self._overlay > 0.1)
            path = self._build_overlay_path(boundary, rows, cols, cp)
            if path:
                p.save()
                p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                p.setBrush(QColor(0, 200, 255, 40))
                p.setPen(QPen(QColor(0, 230, 255), 2))
                p.drawPath(path)
                p.restore()

        # メタターサル中心マーカー
        if self._meta_center is not None:
            cr, cc = self._meta_center
            ir, ic = int(cr), int(cc)
            if 0 <= ir < rows and 0 <= ic < cols:
                x, y = self._grid_to_px(ir, ic)
                p.setPen(QPen(QColor(255, 80, 80), 2))
                cx = int(x + cp // 2)
                cy = int(y + cp // 2)
                p.drawLine(cx - 6, cy, cx + 6, cy)
                p.drawLine(cx, cy - 6, cx, cy + 6)

        # 位置調整タブの角度ガイドライン（踵中心を軸に回転させた基準線）
        if self._angle_guide is not None:
            (r1, c1), (r2, c2) = self._angle_guide
            x1, y1 = self._grid_to_px(r1, c1)
            x2, y2 = self._grid_to_px(r2, c2)
            p.setPen(QPen(QColor(255, 60, 60), 2))
            p.drawLine(int(x1 + cp / 2), int(y1 + cp / 2), int(x2 + cp / 2), int(y2 + cp / 2))

        # グリッド線
        p.setPen(QPen(QColor(50, 50, 50), 1))
        for r in range(rows + 1):
            y = MARGIN_TOP + r * cp
            p.drawLine(MARGIN_LEFT, y, MARGIN_LEFT + cols * cp, y)
        for c in range(cols + 1):
            x = MARGIN_LEFT + c * cp
            p.drawLine(x, MARGIN_TOP, x, MARGIN_TOP + rows * cp)

        # ── 中央縦ライン（col 7 / col 8 境界 = 第2中足骨基準線）──
        x_mid = MARGIN_LEFT + 8 * cp
        cen_pen = QPen(QColor(255, 230, 0, 180), 2)
        cen_pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(cen_pen)
        p.drawLine(x_mid, MARGIN_TOP, x_mid, MARGIN_TOP + rows * cp)
        p.setPen(QPen(QColor(255, 230, 0, 210), 1))
        font.setPointSize(6)
        font.setBold(False)
        p.setFont(font)
        p.drawText(x_mid - 12, MARGIN_TOP - 3, "中央")

        # 選択領域外周ボーダー（隣接セルが未選択の辺にのみ描画）
        if self._sel_mask.any():
            p.setPen(QPen(QColor(0, 230, 255), 2))
            for r in range(rows):
                for c in range(cols):
                    if not self._sel_mask[r, c]:
                        continue
                    x, y = self._grid_to_px(r, c)
                    if r == 0 or not self._sel_mask[r - 1, c]:
                        p.drawLine(x, y, x + cp - 1, y)
                    if r == rows - 1 or not self._sel_mask[r + 1, c]:
                        p.drawLine(x, y + cp - 1, x + cp - 1, y + cp - 1)
                    if c == 0 or not self._sel_mask[r, c - 1]:
                        p.drawLine(x, y, x, y + cp - 1)
                    if c == cols - 1 or not self._sel_mask[r, c + 1]:
                        p.drawLine(x + cp - 1, y, x + cp - 1, y + cp - 1)

        # ミラー選択範囲ボーダー（オレンジ）
        if self._mirror_mask is not None and self._mirror_mask.any():
            p.setPen(QPen(QColor(255, 180, 0), 2))
            for r in range(rows):
                for c in range(cols):
                    if not self._mirror_mask[r, c]:
                        continue
                    x, y = self._grid_to_px(r, c)
                    if r == 0 or not self._mirror_mask[r - 1, c]:
                        p.drawLine(x, y, x + cp - 1, y)
                    if r == rows - 1 or not self._mirror_mask[r + 1, c]:
                        p.drawLine(x, y + cp - 1, x + cp - 1, y + cp - 1)
                    if c == 0 or not self._mirror_mask[r, c - 1]:
                        p.drawLine(x, y, x, y + cp - 1)
                    if c == cols - 1 or not self._mirror_mask[r, c + 1]:
                        p.drawLine(x + cp - 1, y, x + cp - 1, y + cp - 1)

        # 矩形選択プレビュー（Shift+ドラッグ中）
        if self._is_rect_selecting and self._rect_start is not None and self._rect_current is not None:
            r0, c0 = self._rect_start
            r1, c1 = self._rect_current
            rmin, rmax = sorted((r0, r1))
            cmin, cmax = sorted((c0, c1))
            x0, y0 = self._grid_to_px(rmin, cmin)
            x1, y1 = self._grid_to_px(rmax, cmax)
            pen = QPen(QColor(255, 0, 0) if self._rect_subtract else QColor(0, 230, 255), 2, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawRect(x0, y0, x1 + cp - x0, y1 + cp - y0)

        # ── 踵距離ルーラー ──
        font.setPointSize(7)
        font.setBold(False)
        p.setFont(font)
        label_x = CBAR_W + 3
        label_w = MARGIN_LEFT - CBAR_W - 5

        for cm in range(0, 32, 5):
            row_idx = 31 - cm
            _, y = self._grid_to_px(row_idx, 0)
            y_mid = y + cp // 2

            if cm == 0:
                p.setPen(QPen(QColor(140, 140, 140), 1))
                p.drawText(QRect(label_x, y_mid - 6, label_w, 12),
                           Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, "0cm")
            else:
                p.setPen(QPen(QColor(140, 140, 140), 1))
                p.drawText(QRect(label_x, y_mid - 6, label_w, 12),
                           Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                           f"{cm}cm")

        # 列ラベル
        p.setPen(QPen(QColor(160, 160, 160), 1))
        for c in range(0, cols, 2):
            x, y = self._grid_to_px(0, c)
            p.drawText(x + 2, MARGIN_TOP - 4, str(c))

        # タイトル
        font.setPointSize(9)
        font.setBold(True)
        p.setFont(font)
        p.setPen(QPen(QColor(220, 220, 220), 1))
        p.drawText(MARGIN_LEFT, 14, self.title)

        p.end()

    # ─── マウス操作 ───
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            r, c = self._px_to_grid(event.position().x(), event.position().y())
            if 0 <= r < 32 and 0 <= c < 16:
                ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
                shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
                self.cellClicked.emit(r, c)
                if self._select_mode:
                    if shift:
                        # Shift+ドラッグ: 始点〜終点を対角とする矩形を一括選択（Ctrl併用で解除）
                        self._is_rect_selecting = True
                        self._rect_start = (r, c)
                        self._rect_current = (r, c)
                        self._rect_subtract = ctrl
                        self.update()
                    elif ctrl:
                        self._is_deselecting = True
                        self._sel_mask[r, c] = False
                        self.selectionChanged.emit(self._sel_mask.copy())
                        self.update()
                    else:
                        self._is_selecting = True
                        self._sel_mask[r, c] = True
                        self.selectionChanged.emit(self._sel_mask.copy())
                        self.update()

    def mouseMoveEvent(self, event):
        if not self._select_mode:
            return
        r, c = self._px_to_grid(event.position().x(), event.position().y())
        if not (0 <= r < 32 and 0 <= c < 16):
            return
        if self._is_rect_selecting:
            self._rect_current = (r, c)
            self.update()
        elif self._is_selecting:
            self._sel_mask[r, c] = True
            self.selectionChanged.emit(self._sel_mask.copy())
            self.update()
        elif self._is_deselecting:
            self._sel_mask[r, c] = False
            self.selectionChanged.emit(self._sel_mask.copy())
            self.update()

    def mouseReleaseEvent(self, event):
        if self._is_rect_selecting and self._rect_start is not None and self._rect_current is not None:
            r0, c0 = self._rect_start
            r1, c1 = self._rect_current
            rmin, rmax = sorted((r0, r1))
            cmin, cmax = sorted((c0, c1))
            self._sel_mask[rmin:rmax + 1, cmin:cmax + 1] = not self._rect_subtract
            self.selectionChanged.emit(self._sel_mask.copy())
        self._is_selecting = False
        self._is_deselecting = False
        self._is_rect_selecting = False
        self._rect_start = None
        self._rect_current = None
        self.update()
