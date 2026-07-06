"""足底バランス分析"""
import numpy as np

LEFT_ROWS = slice(32, 64)
RIGHT_ROWS = slice(0, 32)


def left_right_ratio(grid: np.ndarray) -> tuple[float, float]:
    """左右荷重比率を返す (left%, right%)"""
    left_sum = float(np.abs(grid[LEFT_ROWS]).sum())
    right_sum = float(np.abs(grid[RIGHT_ROWS]).sum())
    total = left_sum + right_sum
    if total == 0:
        return 50.0, 50.0
    return round(left_sum / total * 100, 1), round(right_sum / total * 100, 1)


def front_back_ratio(foot_grid: np.ndarray) -> tuple[float, float]:
    """前後荷重比率を返す (front%, back%)。入力は32×16（片足）"""
    rows = foot_grid.shape[0]
    mid = rows // 2
    front_sum = float(np.abs(foot_grid[:mid]).sum())
    back_sum = float(np.abs(foot_grid[mid:]).sum())
    total = front_sum + back_sum
    if total == 0:
        return 50.0, 50.0
    return round(front_sum / total * 100, 1), round(back_sum / total * 100, 1)


def contact_area(foot_grid: np.ndarray) -> int:
    """接触セル数（値が負のセル）を返す"""
    return int(np.sum(foot_grid < 0))
