"""3方向保存（元データ/編集後データ/編集履歴）のための純粋関数のユニットテスト（TDD: RED → GREEN）"""
import numpy as np

from main import (
    _build_archive_filename,
    _layer_to_json_dict,
    _validate_save_paths,
    _validate_distinct_paths,
    _reset_marks_dirty,
)
from foot_model import LayerRecord


# ─────────────────────── _validate_distinct_paths ───────────────────────
# (設定ダイアログ用: 空欄は許容し、両方入力済みで同一の場合のみ拒否する)

def test_validate_distinct_paths_allows_both_empty():
    assert _validate_distinct_paths(original_dir="", edited_dir="") is None


def test_validate_distinct_paths_allows_one_empty():
    assert _validate_distinct_paths(original_dir="C:/original", edited_dir="") is None


def test_validate_distinct_paths_rejects_same_non_empty_folder():
    error = _validate_distinct_paths(original_dir="C:/data", edited_dir="C:/data")
    assert error is not None


def test_validate_distinct_paths_accepts_distinct_folders():
    error = _validate_distinct_paths(original_dir="C:/original", edited_dir="C:/edited")
    assert error is None


# ─────────────────────── _validate_save_paths ───────────────────────

def test_validate_save_paths_rejects_empty_original_dir():
    error = _validate_save_paths(original_dir="", edited_dir="C:/edited")
    assert error is not None


def test_validate_save_paths_rejects_empty_edited_dir():
    error = _validate_save_paths(original_dir="C:/original", edited_dir="")
    assert error is not None


def test_validate_save_paths_rejects_same_folder():
    error = _validate_save_paths(original_dir="C:/data", edited_dir="C:/data")
    assert error is not None


def test_validate_save_paths_accepts_distinct_folders():
    error = _validate_save_paths(original_dir="C:/original", edited_dir="C:/edited")
    assert error is None


# ─────────────────────── _build_archive_filename ───────────────────────

def test_build_archive_filename_uses_timestamp_and_original_name():
    name = _build_archive_filename(
        original_filename="rei_1.grd",
        existing_names=set(),
        timestamp="20260707_153000",
    )
    assert name == "20260707_153000_rei_1.grd"


def test_build_archive_filename_avoids_collision_with_counter_suffix():
    first = "20260707_153000_rei_1.grd"
    name = _build_archive_filename(
        original_filename="rei_1.grd",
        existing_names={first},
        timestamp="20260707_153000",
    )
    assert name != first
    assert name == "20260707_153000_rei_1_001.grd"


def test_build_archive_filename_increments_counter_until_free():
    existing = {
        "20260707_153000_rei_1.grd",
        "20260707_153000_rei_1_001.grd",
    }
    name = _build_archive_filename(
        original_filename="rei_1.grd",
        existing_names=existing,
        timestamp="20260707_153000",
    )
    assert name == "20260707_153000_rei_1_002.grd"


# ─────────────────────── _layer_to_json_dict ───────────────────────

def test_layer_to_json_dict_converts_ndarray_params_to_list():
    mask = np.array([[True, False], [False, True]])
    layer = LayerRecord(
        name="手動消去 左足",
        operation="erase",
        params={"foot": "left", "mask": mask},
        enabled=True,
    )
    result = _layer_to_json_dict(layer)
    assert result == {
        "name": "手動消去 左足",
        "operation": "erase",
        "params": {"foot": "left", "mask": [[True, False], [False, True]]},
        "enabled": True,
    }


def test_layer_to_json_dict_passes_through_non_array_params():
    layer = LayerRecord(
        name="自動ノイズ除去",
        operation="noise_removal",
        params={},
        enabled=True,
    )
    result = _layer_to_json_dict(layer)
    assert result == {
        "name": "自動ノイズ除去",
        "operation": "noise_removal",
        "params": {},
        "enabled": True,
    }


# ─────────────────────── _reset_marks_dirty ───────────────────────
# (リセット実行前にレイヤーが存在した場合のみ「未保存の変更あり」とする)

def test_reset_marks_dirty_when_layers_existed():
    assert _reset_marks_dirty(had_layers=True) is True


def test_reset_does_not_mark_dirty_when_no_layers_existed():
    assert _reset_marks_dirty(had_layers=False) is False
