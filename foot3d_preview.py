"""ヒートマップ横の余白に埋め込む小型3Dプレビュー（選択範囲とアーチ調整の適用前形状を表示）。
見た目は計測ソフト Ultra Foot の立体図に揃える。"""
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from foot_model import ARCH_MAX_LOWER_MM
from heatmap_widget import MAX_DEPTH

# 選択色は青い面の上で埋もれない黄色（水色は青と色相が近く見えにくかった）。
# ミラー側はオレンジ（黄色と見分けやすいよう少し赤寄り）
SELECT_RGB = (255 / 255, 240 / 255, 0 / 255)
MIRROR_RGB = (255 / 255, 140 / 255, 0 / 255)
# 見た目は計測ソフト Ultra Foot の立体図に揃える:
# 計測点を交点とする白い網目＋青い面、深いところほど暗く、真上から見下ろす
SURFACE_RGB = (0.0, 0.0, 1.0)
DEPTH_DARKEN = 0.9      # 最も深い点での暗さ（0=変えない、1=真っ黒）
# 陰影は足ごとの最深部ではなく固定の深さを基準にする（同じ深さはどの足でも同じ暗さ）。
# ヒートマップのカラーバーと同じ 20mm で最も暗くなる
SHADE_FULL_DEPTH_MM = MAX_DEPTH
# 高さ方向の表示範囲（どの足でも固定）: 元の最大深さ20mm＋「へこませる」の最大10mm
DISPLAY_MAX_DEPTH_MM = MAX_DEPTH + ARCH_MAX_LOWER_MM
WIRE_COLOR = "white"
WIRE_WIDTH = 0.5

DELTA_MIN_MM = 0.05     # これ未満の変化は「変化なし」とみなす（apply_arch の affected 判定と同じ閾値）
SELECT_ALPHA = 0.75     # 選択色を重ねる割合（下の陰影・網目が透けて見えるよう半透明）
# 持ち上がる範囲の色の濃さは実際の変化量(mm)に比例させる（最大変化を基準にした相対表示にすると、
# 1mmしか上がらなくても5mm上がっても同じ濃さに見え、上がり具合を見誤るため）
DELTA_FULL_MM = 10.0    # この変化量で最も濃くなる
DELTA_MAX_BLEND = 0.6   # 最も濃いときのブレンド率（選択範囲そのものと見分けられるよう SELECT_ALPHA 未満）
INITIAL_ELEV = 90
INITIAL_AZIM_LEFT = -90   # 真上から、つま先が上・かかとが下（ヒートマップと同じ向き）
INITIAL_AZIM_RIGHT = -90
RENDER_INTERVAL_MS = 80  # ドラッグ中の再描画間隔（スロットリング）
BOX_ZOOM = 1.9
# 真上以外（後面など）では網目の線が重なって面の色を隠すため、線を薄くする
TOP_VIEW_MIN_ELEV = 60
SIDE_WIRE_ALPHA = 0.3


class _FillAxes3D(Axes3D):
    """余白（縦長）いっぱいに描くAxes3D。
    標準のAxes3Dは描画領域を正方形に縮めるため縦長の余白では中央の正方形しか使えず、
    はみ出した部分が切れる。正方形化をやめ、代わりに2D投影面の縦範囲を
    表示領域の縦横比に合わせて広げることで、歪ませずに全体を使う。"""

    def apply_aspect(self, position=None):
        if position is None:
            position = self.get_position(original=True)
        self._set_position(position, "active")
        bb = position.transformed(self.get_figure().transSubfigure)
        ratio = bb.height / bb.width if bb.width > 0 else 1.0
        x0, x1 = self.viewLim.intervalx
        self.viewLim.intervaly = (x0 * ratio, x1 * ratio)


def apply_true_scale(ax, rows: int, cols: int, zoom: float = 1.0) -> None:
    """3D軸を実寸 1:1:1（1cm が縦・横・高さのどの方向でも同じ長さ）にする。
    高さの範囲は足ごとに変えず 0〜DISPLAY_MAX_DEPTH_MM に固定し、遠近感で比率を見誤らないよう平行投影にする。
    埋め込み3Dと「3D表示を開く」の両方で使い、補正した高さが見た目どおりに伝わるようにする。
    座標は X=列(1cm)、Y=行(1cm)、Z=深さ(mm)/10=cm。"""
    zspan = DISPLAY_MAX_DEPTH_MM / 10
    ax.set_xlim3d(-0.5, cols - 0.5)
    ax.set_ylim3d(-0.5, rows - 0.5)
    ax.set_zlim3d(-zspan, 0.0)
    ax.set_box_aspect([cols, rows, zspan], zoom=zoom)
    ax.set_proj_type("ortho")


def depth_brightness(depth_mm: np.ndarray) -> np.ndarray:
    """深さ(mm)に応じた青い面の明るさ（0mm=1、SHADE_FULL_DEPTH_MM以上で最も暗い）。"""
    return 1.0 - DEPTH_DARKEN * np.clip(depth_mm / SHADE_FULL_DEPTH_MM, 0.0, 1.0)


def compute_highlight(
    mask: np.ndarray | None,
    delta: np.ndarray | None,
    mirror: bool,
) -> np.ndarray:
    """計測点ごとの強調色と重み (rows, cols, RGBA) を返す（A=面の色に重ねる割合）。
    選択点は黄色（ミラー時オレンジ）で重み SELECT_ALPHA、選択外で実際に高さが変わる点は
    実際の変化量(mm)に比例した重み、それ以外は重み0。"""
    shape = mask.shape if mask is not None else (delta.shape if delta is not None else (32, 16))
    hl = np.zeros((*shape, 4))
    hl[..., :3] = MIRROR_RGB if mirror else SELECT_RGB

    if delta is not None:
        a = np.abs(delta)
        strength = np.clip(a / DELTA_FULL_MM, 0.0, 1.0) * DELTA_MAX_BLEND
        hl[..., 3] = np.where(a >= DELTA_MIN_MM, strength, 0.0)

    if mask is not None:
        hl[..., 3][mask] = SELECT_ALPHA

    return hl


def upsample_surface(grid: np.ndarray) -> np.ndarray:
    """計測点の値をそのまま通る2倍解像度の高さ (2*rows+1, 2*cols+1) を返す。
    奇数番目が計測点（平均しない）、偶数番目は隣り合う計測点の中間（白い網目の線上）と外周。
    面を細かく分けることで、計測点を中心に1マス分の幅で選択色を塗れる。"""
    rows, cols = grid.shape
    p = np.pad(grid, 1, mode="edge")
    fine = np.empty((2 * rows + 1, 2 * cols + 1))
    fine[1::2, 1::2] = grid
    fine[0::2, 1::2] = (p[:-1, 1:-1] + p[1:, 1:-1]) / 2
    fine[1::2, 0::2] = (p[1:-1, :-1] + p[1:-1, 1:]) / 2
    fine[0::2, 0::2] = (p[:-1, :-1] + p[1:, :-1] + p[:-1, 1:] + p[1:, 1:]) / 4
    return fine


class Foot3DPreview(QWidget):
    """片足分の3Dサーフェスを表示する。ドラッグで回転、ダブルクリックで初期視点に戻す。"""

    def __init__(self, initial_azim: int, parent=None):
        super().__init__(parent)
        self._initial_azim = initial_azim
        self._state: tuple | None = None
        self._pending = False
        self._artists: list = []
        self._x_flipped = False
        self.render_count = 0

        self._fig = Figure(facecolor="black")
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setStyleSheet("background-color: black;")
        self.axes = self._fig.add_axes((0, 0, 1, 1), axes_class=_FillAxes3D)
        self.axes.set_facecolor("black")
        self.axes.set_axis_off()
        # 白い網目を常に青い面より手前に描く（mplot3dは描画物単位で前後を決めるため明示する）
        self.axes.computed_zorder = False
        self.axes.view_init(elev=INITIAL_ELEV, azim=initial_azim)
        self._canvas.mpl_connect("button_press_event", self._on_mouse_press)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(RENDER_INTERVAL_MS)
        self._timer.timeout.connect(self.flush)

    def set_data(
        self,
        grid: np.ndarray,
        preview_grid: np.ndarray | None,
        mask: np.ndarray | None,
        mirror: bool,
    ) -> None:
        """表示内容を更新する。再描画は一定間隔にまとめる（ドラッグ選択中の大量呼び出し対策）。"""
        if mask is not None and not mask.any():
            mask = None
        if mask is None:
            preview_grid = None
        if self._state is not None and _same_state(self._state, (grid, preview_grid, mask, mirror)):
            return
        self._state = (grid.copy(), None if preview_grid is None else preview_grid.copy(),
                       None if mask is None else mask.copy(), mirror)
        self._pending = True
        # タイマー稼働中は再始動しない（デバウンスにするとドラッグ中に一度も描かれない）
        if self.isVisible() and not self._timer.isActive():
            self._timer.start()

    def flush(self) -> None:
        """保留中の変更があれば今すぐ描画する。"""
        if not self._pending or self._state is None:
            return
        self._pending = False
        self._render(*self._state)

    def set_view(self, elev: float, azim: float, flip_x: bool) -> None:
        """視点を切り替える（既存3D表示の視点ボタンと同じ定義を受け取る）。
        flip_x は後方視点で足幅方向が左右逆に見えるのを補正する。"""
        self._x_flipped = flip_x
        self.axes.view_init(elev=elev, azim=azim)
        self._apply_xlim()
        if len(self._artists) == 2:
            self._artists[1].set_alpha(_wire_alpha(elev))
        self._canvas.draw_idle()

    def reset_view(self) -> None:
        self.set_view(INITIAL_ELEV, self._initial_azim, flip_x=False)

    def _apply_xlim(self) -> None:
        lo, hi = self.axes.get_xlim3d()
        lo, hi = min(lo, hi), max(lo, hi)
        self.axes.set_xlim3d((hi, lo) if self._x_flipped else (lo, hi))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._pending and not self._timer.isActive():
            self._timer.start()

    def _on_mouse_press(self, event) -> None:
        if event.dblclick:
            self.reset_view()

    def _render(self, grid, preview_grid, mask, mirror) -> None:
        ax = self.axes
        elev, azim = ax.elev, ax.azim
        for artist in self._artists:
            artist.remove()

        rows, cols = grid.shape
        shown = np.flipud(grid if preview_grid is None else preview_grid)  # かかと側が Y=0

        # 青い面: 計測点を通る2倍解像度の面を、深さに応じて暗くして塗る
        fine = upsample_surface(shown)
        fx, fy = np.meshgrid(np.arange(2 * cols + 1) / 2 - 0.5, np.arange(2 * rows + 1) / 2 - 0.5)
        face_depth = -(fine[:-1, :-1] + fine[1:, :-1] + fine[:-1, 1:] + fine[1:, 1:]) / 4
        brightness = depth_brightness(face_depth)
        face_rgb = np.array(SURFACE_RGB) * brightness[..., None]

        # 選択範囲・持ち上がる範囲: 計測点ごとの強調色を、その点を囲む2×2の細かい面に重ねる
        delta = None if preview_grid is None else np.flipud(preview_grid - grid)
        hl = compute_highlight(None if mask is None else np.flipud(mask), delta, mirror)
        hl_fine = hl.repeat(2, axis=0).repeat(2, axis=1)
        weight = hl_fine[..., 3:]
        face_rgb = face_rgb * (1 - weight) + hl_fine[..., :3] * weight

        # plot_surface は面(r, c)の色を facecolors[r][c] から取る（最終行・列は未使用）
        colors = np.ones((2 * rows + 1, 2 * cols + 1, 4))
        colors[:-1, :-1, :3] = face_rgb
        surface = ax.plot_surface(
            fx, fy, fine / 10, facecolors=colors,
            rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, zorder=1,
        )
        # 白い網目: 計測点を交点として直線で結ぶ（既存3D表示・Ultra Foot と同じ形）
        X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
        wire = ax.plot_wireframe(
            X, Y, shown / 10, color=WIRE_COLOR, linewidth=WIRE_WIDTH,
            rstride=1, cstride=1, zorder=2, alpha=_wire_alpha(elev),
        )
        self._artists = [surface, wire]

        apply_true_scale(ax, rows, cols, zoom=BOX_ZOOM)  # zoomは全体の拡大のみで比率は変えない
        self._apply_xlim()
        ax.view_init(elev=elev, azim=azim)
        self._canvas.draw_idle()
        self.render_count += 1


def _wire_alpha(elev: float) -> float:
    return 1.0 if elev >= TOP_VIEW_MIN_ELEV else SIDE_WIRE_ALPHA


def _same_state(old: tuple, new: tuple) -> bool:
    def eq(a, b) -> bool:
        if a is None or b is None:
            return a is b
        return np.array_equal(a, b)
    return all(eq(a, b) for a, b in zip(old[:3], new[:3])) and old[3] == new[3]
