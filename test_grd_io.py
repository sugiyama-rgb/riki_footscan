"""grd_io.PatientInfo の患者情報メタデータ解析のユニットテスト（TDD: RED → GREEN）

実際の旧U2ソフト形式サンプル(rei_1.grd)を調査した結果、行65以降のメタデータのうち
行11=登録シューズサイズ、行20=左足サイズ、行21=右足サイズであることが判明した
（従来の実装では誤って行11=年齢、行20=靴サイズ(EUR)、行21=靴サイズ(JP cm)として
扱っていた）。
"""
from grd_io import _parse_meta, PatientInfo


def _make_meta_lines() -> list[str]:
    lines = [""] * 29
    lines[0] = "_(rei_1)"
    lines[5] = "Japan"
    lines[11] = "25"    # 登録シューズサイズ
    lines[12] = "1"
    lines[13] = "Male"
    lines[17] = "M5UT1-Xb"
    lines[20] = "38.5"  # 左足サイズ
    lines[21] = "39"    # 右足サイズ
    return lines


def test_parse_meta_reads_shoe_size_from_line11():
    p = _parse_meta(_make_meta_lines())
    assert p.shoe_size == "25"


def test_parse_meta_reads_foot_size_left_from_line20():
    p = _parse_meta(_make_meta_lines())
    assert p.foot_size_left == "38.5"


def test_parse_meta_reads_foot_size_right_from_line21():
    p = _parse_meta(_make_meta_lines())
    assert p.foot_size_right == "39"


def test_patient_info_has_no_age_field():
    assert not hasattr(PatientInfo(), "age")
