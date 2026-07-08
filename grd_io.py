"""GRDファイルの読み書き"""
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


GRID_ROWS = 64
GRID_COLS = 16
RIGHT_ROWS = slice(0, 32)   # 行1-32 → 右足
LEFT_ROWS = slice(32, 64)   # 行33-64 → 左足


@dataclass
class PatientInfo:
    name: str = ""
    country: str = ""
    shoe_size: str = ""       # 登録シューズサイズ（行11）
    gender_code: str = ""
    gender: str = ""
    device_id: str = ""
    foot_size_left: str = ""  # 左足サイズ（行20）
    foot_size_right: str = "" # 右足サイズ（行21）
    color_code: str = ""
    set_count: str = ""
    software_version: str = ""
    extra_lines: list = field(default_factory=list)


@dataclass
class GrdData:
    grid: np.ndarray          # shape (64, 16), float64
    patient: PatientInfo
    raw_meta_lines: list      # 行65以降の元テキスト（保存時に使用）


def load(path: str) -> GrdData:
    lines = Path(path).read_bytes().decode("cp932", errors="replace").splitlines()

    grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float64)

    # 行1-64: スペース区切りの浮動小数点数（行番号プレフィックスなし）
    for i in range(min(GRID_ROWS, len(lines))):
        line = lines[i].strip()
        if not line:
            continue
        vals = line.split()
        for col, v in enumerate(vals[:GRID_COLS]):
            try:
                grid[i, col] = float(v)
            except ValueError:
                break

    # 行65以降: メタデータ
    meta_lines: list[str] = lines[GRID_ROWS:] if len(lines) > GRID_ROWS else []

    patient = _parse_meta(meta_lines)
    return GrdData(grid=grid, patient=patient, raw_meta_lines=meta_lines)


def _parse_meta(lines: list[str]) -> PatientInfo:
    """行65以降のメタデータを解析（固定位置ベース）"""
    def get(idx: int) -> str:
        return lines[idx].strip() if idx < len(lines) else ""

    p = PatientInfo()
    if lines:
        # 先頭行: _(name) 形式
        first = get(0)
        if first.startswith("_(") and first.endswith(")"):
            p.name = first[2:-1]
        else:
            p.name = first
        p.country = get(5)
        p.shoe_size = get(11)
        p.gender_code = get(12)
        p.gender = get(13)
        p.device_id = get(17)
        p.foot_size_left = get(20)
        p.foot_size_right = get(21)
        p.color_code = get(22)
        p.set_count = get(24)
        p.software_version = get(28)
    return p


def save(data: GrdData, path: str) -> None:
    lines: list[str] = []

    for row_idx in range(GRID_ROWS):
        vals = " ".join(f"{v:.6f}" for v in data.grid[row_idx])
        lines.append(vals + " ")

    # メタデータをそのまま出力
    for meta_line in data.raw_meta_lines:
        lines.append(meta_line)

    content = "\n".join(lines) + "\n"
    Path(path).write_bytes(content.encode("cp932", errors="replace"))
