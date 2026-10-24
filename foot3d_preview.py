"""ヒートマップの右に置く、両足を1つの画面に表示する3D（選択範囲とアーチ調整の適用前形状を表示）。
見た目は計測ソフト Ultra Foot の立体図に揃える。"""
import matplotlib.cm
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from foot_model import ARCH_MAX_LOWER_MM
from heatmap_widget import MAX_DEPTH

# 選択色は青い面の上で埋もれない黄色（水色は青と色相が近く見えにくかった）。
# ミラー側はオレンジ（黄色と見分けやすいよう少し赤寄り）
SELECT_RGB = (255 / 255, 240 / 255, 0 / 255)
MIRROR_RGB = (255 / 255, 140 / 255, 0 / 255)
# 見た目は計測ソフト Ultra Foot の立体図に揃える:
# 計測点を交点とする白い網目＋青い面、深いところほど暗い
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

FEET_GAP_CM = 2         # 左右の足の間のすき間 (cm)
# 視点 (仰角, 方位角)。グリッドは flipud で Y=0 がかかと。matplotlib のカメラは azim=-90 で -Y（かかと側）にある
INITIAL_VIEW = (30, -90)  # 標準: かかと側から斜め上に見下ろす（つま先が奥＝画面の上に見える）
VIEW_TOP = (90, -90)      # 真上: つま先が上・かかとが下（ヒートマップと同じ向き）
VIEW_BACK = (0, -90)      # 後面: 本当の後ろ（かかと側）から。左足が左に見える（左右反転しない）
VIEW_FRONT = (0, 90)      # 前方: つま先側から（正面から見た像なので左右の足は入れ替わって見える）
VIEW_BOTTOM = (-80, -90)  # 下側: 板の下から見上げる
ROTATE_DEG_PER_HEIGHT = 180  # 画面の高さ分ドラッグしたときに回る角度
# 右ドラッグのズーム: 画面の高さ分ドラッグで e^ZOOM_PER_HEIGHT 倍。全体の拡大率だけを変え、縦横高さの比率（実寸）は保つ
ZOOM_PER_HEIGHT = 2.0
ZOOM_MIN = 0.4
ZOOM_MAX = 5.0
RENDER_INTERVAL_MS = 80  # ドラッグ中の再描画間隔（スロットリング）
# 横からに近い角度（後面など）では網目の線が重なって面の色を隠すため、線を薄くする
SIDE_VIEW_MAX_ELEV = 20
SIDE_WIRE_ALPHA = 0.3

DIFF_MIN_VMAX_MM = 1.0  # 差分表示の色の物差しの最小値 (mm)。変化がほとんど無いときに色が振り切れないようにする

FootInput = tuple  # (grid, preview_grid | None, mask | None, mirror: bool)


class _FillAxes3D(Axes3D):
    """表示領域いっぱいに描くAxes3D。
    標準のAxes3Dは描画領域を正方形に縮めるため縦長・横長の領域では中央の正方形しか使えない。
    正方形化をやめ、代わりに2D投影面の範囲を表示領域の縦横比に合わせて広げることで、歪ませずに全体を使う。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._base_interval = tuple(self.viewLim.intervalx)
        # 平行移動量（表示範囲の幅・高さに対する割合）。Shift＋ドラッグで変わり、拡大率や回転とは独立
        self.pan = [0.0, 0.0]

    def apply_aspect(self, position=None):
        if position is None:
            position = self.get_position(original=True)
        self._set_position(position, "active")
        bb = position.transformed(self.get_figure().transSubfigure)
        ratio = bb.height / bb.width if bb.width > 0 and bb.height > 0 else 1.0
        x0, x1 = self._base_interval
        if ratio >= 1.0:  # 縦長: 縦の範囲を広げる
            xs, ys = (x0, x1), (x0 * ratio, x1 * ratio)
        else:             # 横長: 横の範囲を広げる
            xs, ys = (x0 / ratio, x1 / ratio), (x0, x1)
        dx = -self.pan[0] * (xs[1] - xs[0])
        dy = -self.pan[1] * (ys[1] - ys[0])
        self.viewLim.intervalx = (xs[0] + dx, xs[1] + dx)
        self.viewLim.intervaly = (ys[0] + dy, ys[1] + dy)


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


def _face_mean(fine: np.ndarray) -> np.ndarray:
    """細かい面ごとの4隅の平均。"""
    return (fine[:-1, :-1] + fine[1:, :-1] + fine[:-1, 1:] + fine[1:, 1:]) / 4


def _face_diff(foot: FootInput, base: np.ndarray) -> np.ndarray:
    """補正前（base）からの変化量(mm)を細かい面ごとに返す（正＝持ち上がった）。"""
    grid, preview_grid, _, _ = foot
    shown = grid if preview_grid is None else preview_grid
    return _face_mean(upsample_surface(np.flipud(shown - base)))


def _foot_surface(
    foot: FootInput, x_offset: float, base: np.ndarray | None = None, diff_vmax: float = 1.0,
) -> tuple:
    """片足の細かい面 (x, y, z, 面の色RGB) と網目 (X, Y, Z) を返す（かかと側が Y=0）。
    base を渡すと、面の色を補正前からの変化量の色分け（差分表示）にする。"""
    grid, preview_grid, mask, mirror = foot
    rows, cols = grid.shape
    shown = np.flipud(grid if preview_grid is None else preview_grid)

    fine = upsample_surface(shown)
    fx, fy = np.meshgrid(np.arange(2 * cols + 1) / 2 - 0.5 + x_offset, np.arange(2 * rows + 1) / 2 - 0.5)
    if base is None:
        # 青い面: 計測点を通る2倍解像度の面を、深さに応じて暗くして塗る
        face_rgb = np.array(SURFACE_RGB) * depth_brightness(-_face_mean(fine))[..., None]
    else:
        # 差分表示: 既存の3D表示と同じ plasma の色分け（0〜最大変化）。左右で同じ物差しを使う
        face_rgb = matplotlib.cm.plasma(np.clip(_face_diff(foot, base) / diff_vmax, 0.0, 1.0))[..., :3]

    # 選択範囲・持ち上がる範囲: 計測点ごとの強調色を、その点を囲む2×2の細かい面に重ねる
    delta = None if preview_grid is None else np.flipud(preview_grid - grid)
    hl = compute_highlight(None if mask is None else np.flipud(mask), delta, mirror)
    hl_fine = hl.repeat(2, axis=0).repeat(2, axis=1)
    weight = hl_fine[..., 3:]
    face_rgb = face_rgb * (1 - weight) + hl_fine[..., :3] * weight

    X, Y = np.meshgrid(np.arange(cols) + x_offset, np.arange(rows))
    return (fx, fy, fine / 10, face_rgb), (X, Y, shown / 10)


def build_scene(left: FootInput, right: FootInput, bases: tuple | None = None) -> dict:
    """左足を左・右足を右に FEET_GAP_CM 空けて並べた1つの面と、足ごとの網目を組み立てる。
    片足がもう片足の手前に来る角度でも前後関係が正しくなるよう、面は両足を1つにまとめる
    （間に NaN の列を1本入れて板を分ける。NaN に接する面は描かれない）。
    戻り値: {"x","y","z": 頂点 (2*rows+1, 2*(2*cols+1)+1), "colors": 面の色 RGBA（最終行・列は未使用）,
            "wires": [(X, Y, Z), (X, Y, Z)], "cols": 両足分の横幅(cm), "rows": 縦(cm)}
    bases に (左足の補正前, 右足の補正前) を渡すと差分表示の色分けにする。"""
    rows, cols = left[0].shape
    lbase, rbase = bases if bases is not None else (None, None)
    vmax = 1.0
    if bases is not None:
        vmax = max(float(_face_diff(left, lbase).max()), float(_face_diff(right, rbase).max()), DIFF_MIN_VMAX_MM)
    (lx, ly, lz, lrgb), lwire = _foot_surface(left, 0.0, lbase, vmax)
    (rx, ry, rz, rrgb), rwire = _foot_surface(right, cols + FEET_GAP_CM, rbase, vmax)

    sep_x = np.full((2 * rows + 1, 1), cols - 0.5 + FEET_GAP_CM / 2)
    x = np.hstack([lx, sep_x, rx])
    y = np.hstack([ly, ly[:, :1], ry])
    z = np.hstack([lz, np.full((2 * rows + 1, 1), np.nan), rz])

    # plot_surface は面(r, c)の色を colors[r][c] から取る。色の配列も頂点と同じ形で作り、列がずれないようにする
    colors = np.ones((*z.shape, 4))
    per_foot = 2 * cols + 1
    colors[:-1, : per_foot - 1, :3] = lrgb
    colors[:-1, per_foot + 1: per_foot + per_foot, :3] = rrgb
    return {
        "x": x, "y": y, "z": z, "colors": colors,
        "wires": [lwire, rwire],
        "cols": 2 * cols + FEET_GAP_CM, "rows": rows,
    }


def rotate_view(elev: float, azim: float, dx_ratio: float, dy_ratio: float) -> tuple[float, float]:
    """ドラッグ量（画面の高さに対する比率。matplotlib の座標で y は上が正）から新しい視点を返す。
    横ドラッグで方位角、縦ドラッグで仰角を変える（下へドラッグで手前に倒れる）。
    仰角は 0〜90° に制限し、板の裏側に回り込んで足裏が反転して見えないようにする。
    ただし「下側」ボタンなどで既に0°より下にいるときは、そこから上へ戻れるよう下限を今の仰角にする。"""
    new_azim = azim - dx_ratio * ROTATE_DEG_PER_HEIGHT
    lower = min(0.0, elev)
    new_elev = float(np.clip(elev + dy_ratio * ROTATE_DEG_PER_HEIGHT, lower, 90.0))
    return new_elev, new_azim


class Feet3DView(QWidget):
    """両足を1つの画面に表示する3D。左ドラッグで回転（裏返らない）、右ドラッグでズーム、
    ダブルクリックで標準の視点・大きさに戻す。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state: tuple | None = None
        self._pending = False
        self._surface = None
        self._wires: list = []
        self._drag_from: tuple[float, float] | None = None
        self._drag_button: int | str | None = None  # 1=回転, 3=ズーム, "pan"=平行移動
        self._scene_dims = (32, 2 * 16 + FEET_GAP_CM)  # (rows, cols) 直近に描いた範囲
        self.zoom = 1.0
        self.render_count = 0

        self._fig = Figure(facecolor="black")
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setStyleSheet("background-color: black;")
        self.axes = self._fig.add_axes((0, 0, 1, 1), axes_class=_FillAxes3D)
        self.axes.set_facecolor("black")
        self.axes.set_axis_off()
        # 白い網目を常に青い面より手前に描く（mplot3dは描画物単位で前後を決めるため明示する）
        self.axes.computed_zorder = False
        # matplotlib 標準の回転（自由回転で裏側に回り込む）・パン・ズームは使わず、自前のドラッグ処理にする
        self.axes.disable_mouse_rotation()
        self.axes.view_init(*INITIAL_VIEW, roll=0)
        self._canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self._canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self._canvas.mpl_connect("button_release_event", self._on_mouse_release)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(RENDER_INTERVAL_MS)
        self._timer.timeout.connect(self.flush)

    def set_data(self, left: FootInput, right: FootInput, bases: tuple | None = None) -> None:
        """両足の表示内容を更新する。bases（左右の補正前）を渡すと差分表示の色分けにする。
        再描画は一定間隔にまとめる（ドラッグ選択中の大量呼び出し対策）。"""
        state = (_normalize(left), _normalize(right))
        if (self._state is not None and all(_same_foot(a, b) for a, b in zip(self._state[:2], state))
                and _same_bases(self._state[2], bases)):
            return
        self._state = (*(_copy_foot(f) for f in state),
                       None if bases is None else tuple(b.copy() for b in bases))
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

    def set_view(self, elev: float, azim: float) -> None:
        self.axes.view_init(elev=elev, azim=azim, roll=0)
        self._update_wire_alpha()
        self._canvas.draw_idle()

    def reset_view(self) -> None:
        """標準の視点・大きさ・位置に戻す。"""
        self.zoom = 1.0
        self.axes.pan = [0.0, 0.0]
        self._apply_scale()
        self.set_view(*INITIAL_VIEW)

    def _apply_scale(self) -> None:
        rows, cols = self._scene_dims
        apply_true_scale(self.axes, rows, cols, zoom=self.zoom)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._pending and not self._timer.isActive():
            self._timer.start()

    def _on_mouse_press(self, event) -> None:
        if event.button not in (1, 3):
            return
        if event.button == 1 and event.dblclick:
            self._drag_from = None
            self.reset_view()
            return
        self._drag_from = (event.x, event.y)
        # Shift＋左ドラッグは平行移動
        self._drag_button = "pan" if event.button == 1 and _shift_pressed(event) else event.button

    def _on_mouse_move(self, event) -> None:
        if self._drag_from is None or event.x is None or event.y is None:
            return
        height = max(self._canvas.height() * self._canvas.devicePixelRatioF(), 1.0)
        x0, y0 = self._drag_from
        dx, dy = (event.x - x0) / height, (event.y - y0) / height
        self._drag_from = (event.x, event.y)
        if self._drag_button == "pan":  # Shift＋左ドラッグ: 画面全体を平行移動
            width = max(self._canvas.width() * self._canvas.devicePixelRatioF(), 1.0)
            self.axes.pan[0] += (event.x - x0) / width
            self.axes.pan[1] += (event.y - y0) / height
            self._canvas.draw_idle()
            return
        if self._drag_button == 3:  # 右ドラッグ: 上へで拡大、下へで縮小
            self.zoom = float(np.clip(self.zoom * np.exp(dy * ZOOM_PER_HEIGHT), ZOOM_MIN, ZOOM_MAX))
            self._apply_scale()
            self._canvas.draw_idle()
            return
        elev, azim = rotate_view(self.axes.elev, self.axes.azim, dx, dy)
        self.set_view(elev, azim)

    def _on_mouse_release(self, event) -> None:
        if event.button == self._drag_button or (event.button == 1 and self._drag_button == "pan"):
            self._drag_from = None
            self._drag_button = None

    def _update_wire_alpha(self) -> None:
        alpha = SIDE_WIRE_ALPHA if self.axes.elev < SIDE_VIEW_MAX_ELEV else 1.0
        for wire in self._wires:
            wire.set_alpha(alpha)

    def _render(self, left: FootInput, right: FootInput, bases: tuple | None = None) -> None:
        ax = self.axes
        elev, azim = ax.elev, ax.azim
        for artist in [self._surface, *self._wires]:
            if artist is not None:
                artist.remove()

        scene = build_scene(left, right, bases)
        self._surface = ax.plot_surface(
            scene["x"], scene["y"], scene["z"], facecolors=scene["colors"],
            rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, zorder=1,
        )
        # 白い網目: 計測点を交点として直線で結ぶ（既存3D表示・Ultra Foot と同じ形）
        self._wires = [
            ax.plot_wireframe(X, Y, Z, color=WIRE_COLOR, linewidth=WIRE_WIDTH, rstride=1, cstride=1, zorder=2)
            for X, Y, Z in scene["wires"]
        ]
        # 拡大率1は、どの方位角に回しても両足の板が画面からはみ出さない大きさ（右ドラッグで変更できる）
        self._scene_dims = (scene["rows"], scene["cols"])
        self._apply_scale()
        ax.view_init(elev=elev, azim=azim, roll=0)
        self._update_wire_alpha()
        self._canvas.draw_idle()
        self.render_count += 1


def _normalize(foot: FootInput) -> FootInput:
    grid, preview_grid, mask, mirror = foot
    if mask is not None and not mask.any():
        mask = None
    if mask is None:
        preview_grid = None
    return grid, preview_grid, mask, mirror


def _shift_pressed(event) -> bool:
    """マウス操作時に Shift が押されているか。
    元の Qt のマウスイベント（guiEvent）が持つ修飾キーを優先して見る。matplotlib の event.key は3Dの画面に
    キーボードのフォーカスが無いと空になるため、それだけで判定すると実際の画面では効かない。"""
    gui = getattr(event, "guiEvent", None)
    if gui is not None and hasattr(gui, "modifiers"):
        return bool(gui.modifiers() & Qt.KeyboardModifier.ShiftModifier)
    modifiers = getattr(event, "modifiers", None) or ()
    return "shift" in modifiers or "shift" in (getattr(event, "key", None) or "")


def _same_bases(old: tuple | None, new: tuple | None) -> bool:
    if old is None or new is None:
        return old is new
    return all(np.array_equal(a, b) for a, b in zip(old, new))


def _copy_foot(foot: FootInput) -> FootInput:
    grid, preview_grid, mask, mirror = foot
    return (grid.copy(), None if preview_grid is None else preview_grid.copy(),
            None if mask is None else mask.copy(), mirror)


def _same_foot(old: FootInput, new: FootInput) -> bool:
    def eq(a, b) -> bool:
        if a is None or b is None:
            return a is b
        return np.array_equal(a, b)
    return all(eq(a, b) for a, b in zip(old[:3], new[:3])) and old[3] == new[3]
