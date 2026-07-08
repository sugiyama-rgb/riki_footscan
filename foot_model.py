"""足底データモデルとアルゴリズム"""
import numpy as np
from scipy import ndimage
from dataclasses import dataclass, field
from typing import Optional
import copy

import grd_io


@dataclass
class MetatarsalParams:
    center_row: float = 0.0   # グリッド座標
    center_col: float = 0.0
    height_mm: float = 5.0    # 浅くする量 (mm)
    length_mm: float = 4.0    # 縦方向 (cm単位のグリッド)
    width_mm: float = 2.5     # 横方向 (cm単位)
    angle_deg: float = 0.0    # 回転角
    smoothing: float = 1.0    # ぼかし半径 (グリッド単位)
    front_offset: float = 0.0 # 前方オフセット: 頂点をつま先側へずらす量 (0.0〜0.4)


@dataclass
class ArchParams:
    # 選択済みマスク (32×16 の bool配列)
    mask: Optional[np.ndarray] = None
    height_mm: float = 5.0    # 浅くする量 (mm)
    smoothing: float = 1.5    # ぼかし半径 (グリッド単位)


@dataclass
class LayerRecord:
    name: str
    operation: str  # "noise_removal" | "erase" | "mirror_mask" | "mirror_copy" | "arch" | "metatarsal" | "position_adjust"
    params: dict
    enabled: bool = True


class FootModel:
    def __init__(self, grd: grd_io.GrdData):
        self._original = copy.deepcopy(grd)
        self._grd = copy.deepcopy(grd)
        self._base_grid = grd.grid.copy()
        self._layers: list[LayerRecord] = []
        self._redo_stack: list[LayerRecord] = []

    @property
    def grd(self) -> grd_io.GrdData:
        return self._grd

    @property
    def left_grid(self) -> np.ndarray:
        return self._grd.grid[grd_io.LEFT_ROWS]

    @property
    def right_grid(self) -> np.ndarray:
        return self._grd.grid[grd_io.RIGHT_ROWS]

    @property
    def base_left_grid(self) -> np.ndarray:
        return self._base_grid[grd_io.LEFT_ROWS]

    @property
    def base_right_grid(self) -> np.ndarray:
        return self._base_grid[grd_io.RIGHT_ROWS]

    @property
    def layers(self) -> list[LayerRecord]:
        return self._layers

    def _recompute(self) -> None:
        grid = self._base_grid.copy()
        for layer in self._layers:
            if layer.enabled:
                grid = _apply_operation(grid, layer.operation, layer.params)
        self._grd.grid = grid

    def _add_layer(self, layer: LayerRecord) -> None:
        self._layers.append(layer)
        self._redo_stack.clear()
        self._recompute()

    def toggle_layer(self, index: int) -> None:
        if 0 <= index < len(self._layers):
            self._layers[index].enabled = not self._layers[index].enabled
            self._recompute()

    def set_all_enabled(self, enabled: bool) -> None:
        for layer in self._layers:
            layer.enabled = enabled
        self._recompute()

    def undo(self) -> bool:
        if not self._layers:
            return False
        self._redo_stack.append(self._layers.pop())
        self._recompute()
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        self._layers.append(self._redo_stack.pop())
        self._recompute()
        return True

    def reset(self):
        self._grd.grid = self._base_grid.copy()
        self._layers.clear()
        self._redo_stack.clear()

    # ──────────────────────────────────────────
    # ノイズ除去
    # ──────────────────────────────────────────
    def erase_cells(self, foot: str, mask: np.ndarray) -> int:
        sl = grd_io.LEFT_ROWS if foot == "left" else grd_io.RIGHT_ROWS
        block = self._grd.grid[sl]
        n = int((mask & (block < 0)).sum())
        foot_label = "左足" if foot == "left" else "右足"
        self._add_layer(LayerRecord(
            name=f"手動消去 {foot_label}",
            operation="erase",
            params={"foot": foot, "mask": mask.copy()},
        ))
        return n

    def apply_mirror_mask(self, reference_foot: str, margin: float) -> int:
        ref_sl = grd_io.LEFT_ROWS if reference_foot == "left" else grd_io.RIGHT_ROWS
        if not (self._grd.grid[ref_sl] < 0).any():
            return 0
        target_sl = grd_io.RIGHT_ROWS if reference_foot == "left" else grd_io.LEFT_ROWS
        ref_mask = self._grd.grid[ref_sl] < 0
        if margin > 0:
            ri = int(np.ceil(margin))
            y, x = np.mgrid[-ri:ri + 1, -ri:ri + 1]
            struct = (x ** 2 + y ** 2) <= margin ** 2
            dilated = ndimage.binary_dilation(ref_mask, structure=struct)
        else:
            dilated = ref_mask
        n = int(((self._grd.grid[target_sl] < 0) & ~dilated).sum())
        ref_label = "左足" if reference_foot == "left" else "右足"
        tgt_label = "右足" if reference_foot == "left" else "左足"
        self._add_layer(LayerRecord(
            name=f"対称マスク {ref_label}→{tgt_label} ({margin:.1f}cm)",
            operation="mirror_mask",
            params={"reference_foot": reference_foot, "margin": margin},
        ))
        return n

    def remove_noise(self) -> int:
        test = _apply_operation(self._grd.grid.copy(), "noise_removal", {})
        n = int(((self._grd.grid < 0) & (test >= 0)).sum())
        self._add_layer(LayerRecord(
            name="自動ノイズ除去",
            operation="noise_removal",
            params={},
        ))
        return n

    def mirror_foot(self, source: str) -> None:
        src_label = "左足" if source == "left" else "右足"
        tgt_label = "右足" if source == "left" else "左足"
        self._add_layer(LayerRecord(
            name=f"ミラーコピー {src_label}→{tgt_label}",
            operation="mirror_copy",
            params={"source": source},
        ))

    # ──────────────────────────────────────────
    # 位置調整（左右・前後オフセット＋任意角度）
    # ──────────────────────────────────────────
    def preview_position_adjust(self, foot: str, dx_cm: float, dy_cm: float, angle_deg: float = 0.0) -> dict:
        """モデルを変更せずに、適用した場合の失われるセル数だけを試算する"""
        sl = grd_io.LEFT_ROWS if foot == "left" else grd_io.RIGHT_ROWS
        before_cells = int((self._grd.grid[sl] < 0).sum())
        params = {"foot": foot, "dx_cm": dx_cm, "dy_cm": dy_cm, "angle_deg": angle_deg}
        test_grid = _apply_operation(self._grd.grid.copy(), "position_adjust", params)
        after_cells = int((test_grid[sl] < 0).sum())
        return {
            "before_cells": before_cells,
            "after_cells": after_cells,
            "lost_cells": max(0, before_cells - after_cells),
        }

    def apply_position_adjust(self, foot: str, dx_cm: float, dy_cm: float, angle_deg: float = 0.0) -> dict:
        sl = grd_io.LEFT_ROWS if foot == "left" else grd_io.RIGHT_ROWS
        before_cells = int((self._grd.grid[sl] < 0).sum())

        foot_label = "左足" if foot == "left" else "右足"
        new_layer = LayerRecord(
            name=f"位置調整 {foot_label} (左右{dx_cm:+.1f}cm/前後{dy_cm:+.1f}cm/角度{angle_deg:+.1f}°)",
            operation="position_adjust",
            params={"foot": foot, "dx_cm": dx_cm, "dy_cm": dy_cm, "angle_deg": angle_deg},
        )

        # 同じ足に対する position_adjust レイヤーが既にある場合は追加ではなく置き換える。
        # 追加してしまうと、適用のたびに前回の補間結果へさらに補間をかける「多重ボケ」が
        # 蓄積するため、常に _base_grid からの1回分の補間に保つ。
        existing_idx = next(
            (i for i, layer in enumerate(self._layers)
             if layer.operation == "position_adjust" and layer.params["foot"] == foot),
            None,
        )
        if existing_idx is not None:
            self._layers[existing_idx] = new_layer
            self._redo_stack.clear()
            self._recompute()
        else:
            self._add_layer(new_layer)

        after_cells = int((self._grd.grid[sl] < 0).sum())
        return {
            "before_cells": before_cells,
            "after_cells": after_cells,
            "lost_cells": max(0, before_cells - after_cells),
        }

    # ──────────────────────────────────────────
    # 内側縦アーチ調整
    # ──────────────────────────────────────────
    def apply_arch(self, foot: str, params: ArchParams) -> dict | None:
        if params.mask is None:
            return None
        sl = grd_io.LEFT_ROWS if foot == "left" else grd_io.RIGHT_ROWS
        prev = self._grd.grid[sl].copy()
        foot_label = "左足" if foot == "left" else "右足"
        self._add_layer(LayerRecord(
            name=f"アーチ調整 {foot_label} {params.height_mm:.1f}mm",
            operation="arch",
            params={
                "foot": foot,
                "mask": params.mask.copy(),
                "height_mm": params.height_mm,
                "smoothing": params.smoothing,
            },
        ))
        diff = self._grd.grid[sl] - prev
        return {
            "set_mm": params.height_mm,
            "actual_max": float(np.max(diff)),
            "affected": int(np.sum(diff > 0.05)),
        }

    # ──────────────────────────────────────────
    # メタターサルサポート
    # ──────────────────────────────────────────
    def apply_metatarsal(self, foot: str, params: MetatarsalParams) -> dict:
        sl = grd_io.LEFT_ROWS if foot == "left" else grd_io.RIGHT_ROWS
        prev = self._grd.grid[sl].copy()
        foot_label = "左足" if foot == "left" else "右足"
        self._add_layer(LayerRecord(
            name=f"メタターサル {foot_label} {params.height_mm:.1f}mm",
            operation="metatarsal",
            params={
                "foot": foot,
                "center_row": params.center_row,
                "center_col": params.center_col,
                "height_mm": params.height_mm,
                "length_mm": params.length_mm,
                "width_mm": params.width_mm,
                "angle_deg": params.angle_deg,
                "smoothing": params.smoothing,
                "front_offset": params.front_offset,
            },
        ))
        diff = self._grd.grid[sl] - prev
        return {
            "set_mm": params.height_mm,
            "actual_max": float(np.max(diff)),
            "affected": int(np.sum(diff > 0.05)),
        }


# ──────────────────────────────────────────────
# レイヤー適用エンジン
# ──────────────────────────────────────────────
def _apply_operation(grid: np.ndarray, operation: str, params: dict) -> np.ndarray:
    result = grid.copy()

    if operation == "noise_removal":
        for sl in (grd_io.LEFT_ROWS, grd_io.RIGHT_ROWS):
            block = result[sl].copy()
            mask = block < 0
            labeled, n = ndimage.label(mask)
            if n == 0:
                continue
            sizes = ndimage.sum(mask, labeled, range(1, n + 1))
            largest = int(np.argmax(sizes)) + 1
            noise_mask = mask & (labeled != largest)
            block[noise_mask] = 0.0
            result[sl] = block

    elif operation == "erase":
        foot = params["foot"]
        mask = params["mask"]
        sl = grd_io.LEFT_ROWS if foot == "left" else grd_io.RIGHT_ROWS
        result[sl][mask] = 0.0

    elif operation == "mirror_mask":
        reference_foot = params["reference_foot"]
        margin = params["margin"]
        ref_sl = grd_io.LEFT_ROWS if reference_foot == "left" else grd_io.RIGHT_ROWS
        target_sl = grd_io.RIGHT_ROWS if reference_foot == "left" else grd_io.LEFT_ROWS
        ref_mask = result[ref_sl] < 0
        if not ref_mask.any():
            return result
        if margin > 0:
            ri = int(np.ceil(margin))
            y, x = np.mgrid[-ri:ri + 1, -ri:ri + 1]
            struct = (x ** 2 + y ** 2) <= margin ** 2
            dilated = ndimage.binary_dilation(ref_mask, structure=struct)
        else:
            dilated = ref_mask
        target_block = result[target_sl].copy()
        target_block[(target_block < 0) & ~dilated] = 0.0
        result[target_sl] = target_block

    elif operation == "mirror_copy":
        source = params["source"]
        src_sl = grd_io.LEFT_ROWS if source == "left" else grd_io.RIGHT_ROWS
        tgt_sl = grd_io.RIGHT_ROWS if source == "left" else grd_io.LEFT_ROWS
        result[tgt_sl] = np.fliplr(result[src_sl].copy())

    elif operation == "position_adjust":
        foot = params["foot"]
        sl = grd_io.LEFT_ROWS if foot == "left" else grd_io.RIGHT_ROWS
        block = result[sl]
        dx = params.get("dx_cm", 0.0)
        dy = params.get("dy_cm", 0.0)
        angle = params.get("angle_deg", 0.0)
        if dx or dy or angle:
            theta = np.deg2rad(angle)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            # 出力座標→入力座標への逆写像行列（ブロック中心を軸に回転させ、その後に平行移動する）
            inv_rot = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
            # 回転中心は踵中心（row = 各足ブロックの最終行 = 踵側、col = 幅方向の中央）
            center = np.array([block.shape[0] - 1, (block.shape[1] - 1) / 2.0])
            shift_vec = np.array([dy, dx])
            offset = center - inv_rot @ (center + shift_vec)
            block = ndimage.affine_transform(
                block, matrix=inv_rot, offset=offset,
                order=1, mode="constant", cval=0.0,
            )
        result[sl] = block

    elif operation == "arch":
        foot = params["foot"]
        sl = grd_io.LEFT_ROWS if foot == "left" else grd_io.RIGHT_ROWS
        mask_f = params["mask"].astype(np.float64)
        smooth_mask = ndimage.gaussian_filter(mask_f, sigma=params["smoothing"])
        smooth_mask = np.clip(smooth_mask, 0, 1)
        delta = smooth_mask * params["height_mm"]
        block = result[sl].copy()
        result[sl] = np.minimum(block + delta, 0.0)

    elif operation == "metatarsal":
        foot = params["foot"]
        sl = grd_io.LEFT_ROWS if foot == "left" else grd_io.RIGHT_ROWS
        block = result[sl].copy()
        bump = _make_teardrop_bump(
            shape=block.shape,
            center_rc=(params["center_row"], params["center_col"]),
            height=params["height_mm"],
            length=params["length_mm"],
            width=params["width_mm"],
            angle_deg=params["angle_deg"],
            smoothing=params["smoothing"],
            front_offset=params.get("front_offset", 0.0),
        )
        result[sl] = np.minimum(block + bump, 0.0)

    return result


# ──────────────────────────────────────────────
# 涙型バンプ生成
# ──────────────────────────────────────────────
def preview_arch_max(mask: np.ndarray, height_mm: float, smoothing: float) -> float:
    """スムージング後のアーチ調整理論最大値（適用前プレビュー用）"""
    smooth_mask = ndimage.gaussian_filter(mask.astype(np.float64), sigma=smoothing)
    return float(np.max(np.clip(smooth_mask, 0, 1))) * height_mm


def _make_teardrop_bump(
    shape: tuple,
    center_rc: tuple,
    height: float,
    length: float,
    width: float,
    angle_deg: float,
    smoothing: float,
    front_offset: float = 0.0,
) -> np.ndarray:
    """
    涙型（縦長の楕円＋先端が尖る）のなだらかな丘を生成。
    返り値は shape と同じサイズの float64 配列 (値は 0〜height)
    """
    rows, cols = shape
    cr, cc = center_rc
    angle_rad = np.deg2rad(angle_deg)

    r_idx, c_idx = np.mgrid[0:rows, 0:cols]
    dr = r_idx - cr
    dc = c_idx - cc

    # 回転
    dr_rot = dr * np.cos(angle_rad) + dc * np.sin(angle_rad)
    dc_rot = -dr * np.sin(angle_rad) + dc * np.cos(angle_rad)

    a = length * 0.5   # つま先側（前方）の半軸: 固定
    b = width / 2.0
    # front_offset > 0: 踵側の半軸だけを伸ばし、クリック位置が形状の前端（つま先寄り）に見える
    # shift方式は楕円前端にクリックが寄りすぎてひし形になるため、非対称半軸方式を採用
    a_back = a * (1.0 + 2.0 * front_offset)  # 踵側（後方）の半軸: front_offsetで伸長

    # 非対称楕円距離: つま先側は a、踵側は a_back を使用
    dr_sq = np.where(dr_rot <= 0, (dr_rot / a) ** 2, (dr_rot / a_back) ** 2)
    ellipse_dist = np.sqrt(dr_sq + (dc_rot / (b + 1e-9)) ** 2)

    # 涙型: 踵方向（正のdr_rot）に向かって先端を尖らせる（a_backで正規化）
    taper = np.where(dr_rot > 0, 1.0 + 0.4 * (dr_rot / a_back), 1.0)
    teardrop_dist = ellipse_dist * taper

    # コサイン形状で丘を作る
    inside = teardrop_dist < 1.0
    bump = np.where(inside, height * 0.5 * (1 + np.cos(np.pi * teardrop_dist)), 0.0)

    # 境界のなだらかさを追加
    if smoothing > 0:
        bump = ndimage.gaussian_filter(bump, sigma=smoothing)

    return bump.astype(np.float64)
