"""足底スキャンエディタ - メインウィンドウ"""
import sys
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGroupBox, QPushButton, QLabel, QSlider, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QTabWidget, QScrollArea, QLineEdit,
    QComboBox, QCheckBox, QSplitter, QStatusBar, QFormLayout,
    QListWidget, QListWidgetItem, QDialog,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

import json
import copy
from platformdirs import user_config_dir

import grd_io
from foot_model import FootModel, ArchParams, MetatarsalParams, preview_arch_max
from heatmap_widget import HeatmapWidget


_DEFAULT_SETTINGS: dict = {
    "arch": {"height_mm": 5.0, "smoothing": 1.5},
    "metatarsal": {
        "height_mm": 5.0,
        "length_mm": 4.0,
        "width_mm": 2.5,
        "angle_deg": 0.0,
        "smoothing": 1.0,
        "front_offset": 0.0,
    },
}


def _load_settings() -> dict:
    path = Path(user_config_dir("riki_footscan")) / "settings.json"
    if not path.exists():
        return copy.deepcopy(_DEFAULT_SETTINGS)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        result = copy.deepcopy(_DEFAULT_SETTINGS)
        for section, values in data.items():
            if section in result and isinstance(values, dict):
                result[section].update(values)
        return result
    except Exception:
        return copy.deepcopy(_DEFAULT_SETTINGS)


def _save_settings(values: dict) -> None:
    path = Path(user_config_dir("riki_footscan")) / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(values, f, indent=2, ensure_ascii=False)


class SettingsDialog(QDialog):
    def __init__(self, defaults: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("設定")
        self.setMinimumWidth(340)
        layout = QVBoxLayout(self)

        arch_box = QGroupBox("アーチ調整 初期値")
        arch_form = QFormLayout(arch_box)

        self._arch_height = QDoubleSpinBox()
        self._arch_height.setRange(1.0, 20.0)
        self._arch_height.setSingleStep(0.5)
        self._arch_height.setDecimals(1)
        self._arch_height.setSuffix(" mm")
        self._arch_height.setValue(float(defaults["arch"]["height_mm"]))
        arch_form.addRow("調整量:", self._arch_height)

        self._arch_smooth = QDoubleSpinBox()
        self._arch_smooth.setRange(0.5, 5.0)
        self._arch_smooth.setSingleStep(0.5)
        self._arch_smooth.setSuffix(" cm")
        self._arch_smooth.setValue(defaults["arch"]["smoothing"])
        arch_form.addRow("スムージング幅:", self._arch_smooth)

        layout.addWidget(arch_box)

        meta_box = QGroupBox("メタターサルサポート 初期値")
        meta_form = QFormLayout(meta_box)

        self._meta_height = QDoubleSpinBox()
        self._meta_height.setRange(0.5, 20.0)
        self._meta_height.setSingleStep(0.5)
        self._meta_height.setSuffix(" mm")
        self._meta_height.setValue(defaults["metatarsal"]["height_mm"])
        meta_form.addRow("高さ:", self._meta_height)

        self._meta_length = QDoubleSpinBox()
        self._meta_length.setRange(1.0, 10.0)
        self._meta_length.setSingleStep(0.5)
        self._meta_length.setSuffix(" cm")
        self._meta_length.setValue(defaults["metatarsal"]["length_mm"])
        meta_form.addRow("長さ:", self._meta_length)

        self._meta_width = QDoubleSpinBox()
        self._meta_width.setRange(0.5, 6.0)
        self._meta_width.setSingleStep(0.5)
        self._meta_width.setSuffix(" cm")
        self._meta_width.setValue(defaults["metatarsal"]["width_mm"])
        meta_form.addRow("幅:", self._meta_width)

        self._meta_angle = QDoubleSpinBox()
        self._meta_angle.setRange(-90.0, 90.0)
        self._meta_angle.setSingleStep(5.0)
        self._meta_angle.setSuffix(" °")
        self._meta_angle.setValue(defaults["metatarsal"]["angle_deg"])
        meta_form.addRow("角度:", self._meta_angle)

        self._meta_smooth = QDoubleSpinBox()
        self._meta_smooth.setRange(0.0, 3.0)
        self._meta_smooth.setSingleStep(0.5)
        self._meta_smooth.setValue(defaults["metatarsal"]["smoothing"])
        meta_form.addRow("スムージング:", self._meta_smooth)

        self._meta_front_offset = QDoubleSpinBox()
        self._meta_front_offset.setRange(0.00, 0.40)
        self._meta_front_offset.setSingleStep(0.05)
        self._meta_front_offset.setDecimals(2)
        self._meta_front_offset.setValue(defaults["metatarsal"]["front_offset"])
        meta_form.addRow("前方オフセット:", self._meta_front_offset)

        layout.addWidget(meta_box)

        btn_row = QHBoxLayout()
        btn_save = QPushButton("保存して閉じる")
        btn_save.setDefault(True)
        btn_cancel = QPushButton("キャンセル")
        btn_save.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_save)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

    def get_values(self) -> dict:
        return {
            "arch": {
                "height_mm": self._arch_height.value(),
                "smoothing": self._arch_smooth.value(),
            },
            "metatarsal": {
                "height_mm": self._meta_height.value(),
                "length_mm": self._meta_length.value(),
                "width_mm": self._meta_width.value(),
                "angle_deg": self._meta_angle.value(),
                "smoothing": self._meta_smooth.value(),
                "front_offset": self._meta_front_offset.value(),
            },
        }


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("足底スキャンエディタ")
        self.setMinimumSize(1100, 800)
        self._model: FootModel | None = None
        self._current_foot = "left"
        self._arch_params = ArchParams()
        self._meta_params = MetatarsalParams()
        self._erase_select_mode = False

        self._3d_dialog = None
        self._3d_canvas = None
        self._3d_axes: list = []
        self._3d_wireframes: list = []
        self._3d_x_flip: list = [False, False]
        self._3d_y_flip: list = [False, False]

        self._defaults = _load_settings()
        self._build_ui()
        self._update_status("GRDファイルを開いてください")

    # ──────────────────────────────────────────
    # UI構築
    # ──────────────────────────────────────────
    def _build_ui(self):
        menu_bar = self.menuBar()
        settings_menu = menu_bar.addMenu("設定")
        act_settings = settings_menu.addAction("設定を開く...")
        act_settings.triggered.connect(self._open_settings)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── 左: ヒートマップエリア
        map_area = QWidget()
        map_layout = QVBoxLayout(map_area)
        map_layout.setSpacing(4)

        foot_label = QLabel("左足 / 右足")
        foot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(10)
        foot_label.setFont(font)
        map_layout.addWidget(foot_label)

        maps_row = QHBoxLayout()
        self._hm_left = HeatmapWidget("左足")
        self._hm_right = HeatmapWidget("右足")
        self._hm_left.cellClicked.connect(lambda r, c: self._on_cell_clicked("left", r, c))
        self._hm_right.cellClicked.connect(lambda r, c: self._on_cell_clicked("right", r, c))
        self._hm_left.selectionChanged.connect(lambda mask: self._on_selection_changed("left", mask))
        self._hm_right.selectionChanged.connect(lambda mask: self._on_selection_changed("right", mask))
        maps_row.addWidget(self._hm_left)
        maps_row.addSpacing(12)
        maps_row.addWidget(self._hm_right)
        map_layout.addLayout(maps_row)

        splitter.addWidget(map_area)

        # ── 右: ツールパネル（スクロール対応）
        tool_scroll = QScrollArea()
        tool_scroll.setWidgetResizable(True)
        tool_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        tool_scroll.setMinimumWidth(300)
        tool_panel = QWidget()
        tool_scroll.setWidget(tool_panel)
        splitter.addWidget(tool_scroll)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([780, 340])

        tool_layout = QVBoxLayout(tool_panel)
        tool_layout.setSpacing(6)

        # ファイル操作
        file_box = QGroupBox("ファイル")
        file_form = QVBoxLayout(file_box)
        btn_open = QPushButton("開く (GRD)")
        btn_open.clicked.connect(self._open_file)
        btn_save = QPushButton("上書き保存")
        btn_save.clicked.connect(self._save_file)
        btn_saveas = QPushButton("名前をつけて保存")
        btn_saveas.clicked.connect(self._save_as_file)
        for b in (btn_open, btn_save, btn_saveas):
            file_form.addWidget(b)
        tool_layout.addWidget(file_box)

        # 編集操作
        edit_box = QGroupBox("編集")
        edit_form = QHBoxLayout(edit_box)
        btn_undo = QPushButton("元に戻す")
        btn_undo.clicked.connect(self._undo)
        btn_reset = QPushButton("全リセット")
        btn_reset.clicked.connect(self._reset)
        edit_form.addWidget(btn_undo)
        edit_form.addWidget(btn_reset)
        tool_layout.addWidget(edit_box)

        # 表示
        view_box = QGroupBox("表示")
        view_form = QVBoxLayout(view_box)
        btn_3d = QPushButton("3D表示を開く")
        btn_3d.clicked.connect(self._show_3d_view)
        view_form.addWidget(btn_3d)
        tool_layout.addWidget(view_box)

        # 調整レイヤー
        layer_box = QGroupBox("調整レイヤー")
        layer_layout = QVBoxLayout(layer_box)
        layer_layout.setSpacing(2)
        layer_layout.setContentsMargins(6, 6, 6, 6)
        self._btn_toggle_all = QPushButton("補正 全OFF")
        self._btn_toggle_all.setCheckable(True)
        self._btn_toggle_all.setStyleSheet(
            "QPushButton:checked { background-color: #884400; color: white; }"
        )
        self._btn_toggle_all.toggled.connect(self._on_toggle_all_layers)
        layer_layout.addWidget(self._btn_toggle_all)

        self._btn_diff = QPushButton("差分表示")
        self._btn_diff.setCheckable(True)
        self._btn_diff.setStyleSheet(
            "QPushButton:checked { background-color: #004422; color: #00cc66; }"
        )
        self._btn_diff.toggled.connect(self._on_diff_toggled)
        layer_layout.addWidget(self._btn_diff)

        self._layer_list = QListWidget()
        self._layer_list.setMaximumHeight(110)
        self._layer_list.setStyleSheet("QListWidget { font-size: 11px; }")
        self._layer_list.itemChanged.connect(self._on_layer_item_changed)
        layer_layout.addWidget(self._layer_list)
        tool_layout.addWidget(layer_box)

        # タブ
        tabs = QTabWidget()
        tabs.addTab(self._build_noise_tab(), "ノイズ除去")
        tabs.addTab(self._build_arch_tab(), "アーチ調整")
        tabs.addTab(self._build_meta_tab(), "メタターサル")
        tabs.addTab(self._build_patient_tab(), "患者情報")
        tabs.currentChanged.connect(self._on_tab_changed)
        self._tabs = tabs
        tool_layout.addWidget(tabs, stretch=1)

        tool_layout.addStretch()

        # ステータスバー
        self.setStatusBar(QStatusBar())

    # ─── ノイズ除去タブ ───
    def _build_noise_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(6)

        # 自動ノイズ除去
        auto_box = QGroupBox("自動ノイズ除去")
        auto_layout = QVBoxLayout(auto_box)
        auto_layout.setSpacing(4)
        lbl_auto = QLabel("足形状から孤立したピンデータを自動で除去します。")
        lbl_auto.setWordWrap(True)
        auto_layout.addWidget(lbl_auto)
        btn_auto = QPushButton("ノイズ除去を実行")
        btn_auto.clicked.connect(self._run_noise_removal)
        auto_layout.addWidget(btn_auto)
        layout.addWidget(auto_box)

        # ① 手動消去
        erase_box = QGroupBox("① 手動消去")
        erase_layout = QVBoxLayout(erase_box)
        erase_layout.setSpacing(4)
        erase_layout.addWidget(QLabel("対象足:"))
        self._erase_foot_combo = QComboBox()
        self._erase_foot_combo.addItems(["左足", "右足"])
        erase_layout.addWidget(self._erase_foot_combo)
        lbl_erase = QLabel("ヒートマップ上でドラッグして消去する領域を選択")
        lbl_erase.setWordWrap(True)
        erase_layout.addWidget(lbl_erase)
        self._btn_erase_select = QPushButton("選択モード ON")
        self._btn_erase_select.setCheckable(True)
        self._btn_erase_select.toggled.connect(self._toggle_erase_select)
        erase_layout.addWidget(self._btn_erase_select)
        btn_clear_erase = QPushButton("選択をクリア")
        btn_clear_erase.clicked.connect(self._clear_erase_selection)
        erase_layout.addWidget(btn_clear_erase)
        btn_apply_erase = QPushButton("選択セルを消去")
        btn_apply_erase.clicked.connect(self._apply_erase)
        erase_layout.addWidget(btn_apply_erase)
        layout.addWidget(erase_box)

        # ② 左右対称マスク
        mirror_box = QGroupBox("② 左右対称マスク")
        mirror_layout = QVBoxLayout(mirror_box)
        mirror_layout.setSpacing(4)
        mirror_layout.addWidget(QLabel("正常な足を参照に使用:"))
        self._mirror_ref_combo = QComboBox()
        self._mirror_ref_combo.addItems(["左足を参照（右足を整形）", "右足を参照（左足を整形）"])
        mirror_layout.addWidget(self._mirror_ref_combo)
        margin_row = QHBoxLayout()
        margin_row.addWidget(QLabel("余白 (cm):"))
        self._mirror_margin = QDoubleSpinBox()
        self._mirror_margin.setRange(0.0, 5.0)
        self._mirror_margin.setSingleStep(0.5)
        self._mirror_margin.setValue(1.0)
        margin_row.addWidget(self._mirror_margin)
        mirror_layout.addLayout(margin_row)
        btn_apply_mirror = QPushButton("対称マスクを適用")
        btn_apply_mirror.clicked.connect(self._apply_mirror_mask)
        mirror_layout.addWidget(btn_apply_mirror)
        layout.addWidget(mirror_box)

        # ③ ミラーコピー
        mirror_copy_box = QGroupBox("③ ミラーコピー")
        mirror_copy_layout = QVBoxLayout(mirror_copy_box)
        mirror_copy_layout.setSpacing(4)
        lbl_mc = QLabel("片方の足を左右反転して反対側の足に上書きコピーします。")
        lbl_mc.setWordWrap(True)
        mirror_copy_layout.addWidget(lbl_mc)
        self._mirror_copy_combo = QComboBox()
        self._mirror_copy_combo.addItems(["左足 → 右足にコピー", "右足 → 左足にコピー"])
        mirror_copy_layout.addWidget(self._mirror_copy_combo)
        btn_apply_mirror_copy = QPushButton("ミラーコピーを実行")
        btn_apply_mirror_copy.clicked.connect(self._apply_mirror_copy)
        mirror_copy_layout.addWidget(btn_apply_mirror_copy)
        layout.addWidget(mirror_copy_box)

        step_row = QHBoxLayout()
        btn_nu = QPushButton("↩ 戻す")
        btn_nu.clicked.connect(self._undo)
        btn_nr = QPushButton("↪ 進む")
        btn_nr.clicked.connect(self._redo)
        step_row.addWidget(btn_nu)
        step_row.addWidget(btn_nr)
        layout.addLayout(step_row)

        layout.addStretch()
        return w

    # ─── アーチ調整タブ ───
    def _build_arch_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(8)

        # 対象足
        layout.addWidget(QLabel("対象足:"))
        self._arch_foot_combo = QComboBox()
        self._arch_foot_combo.addItems(["左足", "右足"])
        layout.addWidget(self._arch_foot_combo)

        layout.addWidget(QLabel("① ヒートマップ上でアーチ領域を\n   ドラッグして選択してください"))

        self._btn_arch_select = QPushButton("領域選択モード ON")
        self._btn_arch_select.setCheckable(True)
        self._btn_arch_select.toggled.connect(self._toggle_arch_select)
        layout.addWidget(self._btn_arch_select)

        btn_clear_sel = QPushButton("選択をクリア")
        btn_clear_sel.clicked.connect(self._clear_arch_selection)
        layout.addWidget(btn_clear_sel)

        # 高さスライダー（内部値×2: range 2〜40 = 1.0〜20.0mm）
        layout.addWidget(QLabel("② 調整量 (0.5〜20 mm):"))
        row = QHBoxLayout()
        self._arch_slider = QSlider(Qt.Orientation.Horizontal)
        self._arch_slider.setRange(2, 40)
        self._arch_slider.setValue(int(self._defaults["arch"]["height_mm"] * 2))
        self._arch_slider.valueChanged.connect(self._on_arch_slider)
        self._arch_spin = QDoubleSpinBox()
        self._arch_spin.setRange(0.5, 20.0)
        self._arch_spin.setSingleStep(0.5)
        self._arch_spin.setDecimals(1)
        self._arch_spin.setValue(self._defaults["arch"]["height_mm"])
        self._arch_spin.valueChanged.connect(lambda v: self._arch_slider.setValue(int(v * 2)))
        self._arch_slider.valueChanged.connect(lambda v: self._arch_spin.setValue(v / 2.0))
        row.addWidget(self._arch_slider)
        row.addWidget(self._arch_spin)
        layout.addLayout(row)

        # スムージング
        layout.addWidget(QLabel("スムージング幅 (cm):"))
        self._arch_smooth = QDoubleSpinBox()
        self._arch_smooth.setRange(0.5, 5.0)
        self._arch_smooth.setSingleStep(0.5)
        self._arch_smooth.setValue(self._defaults["arch"]["smoothing"])
        self._arch_smooth.valueChanged.connect(lambda _: self._update_arch_preview_label())
        layout.addWidget(self._arch_smooth)

        btn_apply = QPushButton("③ アーチ調整を適用")
        btn_apply.clicked.connect(self._apply_arch)
        layout.addWidget(btn_apply)

        self._arch_mirror_check = QCheckBox("反対足にも適用（ミラー）")
        self._arch_mirror_check.toggled.connect(self._on_arch_mirror_toggled)
        layout.addWidget(self._arch_mirror_check)

        self._arch_preview_label = QLabel("")
        self._arch_preview_label.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        layout.addWidget(self._arch_preview_label)

        self._arch_stats_label = QLabel("")
        self._arch_stats_label.setWordWrap(True)
        self._arch_stats_label.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        layout.addWidget(self._arch_stats_label)

        btn_reset_arch = QPushButton("設定値を初期化")
        btn_reset_arch.clicked.connect(self._reset_arch_params)
        layout.addWidget(btn_reset_arch)

        step_row = QHBoxLayout()
        btn_undo = QPushButton("↩ 戻す")
        btn_undo.clicked.connect(self._undo)
        btn_redo = QPushButton("↪ 進む")
        btn_redo.clicked.connect(self._redo)
        step_row.addWidget(btn_undo)
        step_row.addWidget(btn_redo)
        layout.addLayout(step_row)

        layout.addStretch()
        return w

    # ─── メタターサルタブ ───
    def _build_meta_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(6)

        layout.addWidget(QLabel("対象足:"))
        self._meta_foot_combo = QComboBox()
        self._meta_foot_combo.addItems(["左足", "右足"])
        layout.addWidget(self._meta_foot_combo)

        layout.addWidget(QLabel("① ヒートマップをクリックして\n   配置位置を指定してください"))

        form = QFormLayout()
        self._meta_height = QDoubleSpinBox()
        self._meta_height.setRange(0.5, 20.0)
        self._meta_height.setSingleStep(0.5)
        self._meta_height.setValue(self._defaults["metatarsal"]["height_mm"])
        self._meta_height.valueChanged.connect(self._update_meta_preview)

        self._meta_length = QDoubleSpinBox()
        self._meta_length.setRange(1.0, 10.0)
        self._meta_length.setSingleStep(0.5)
        self._meta_length.setValue(self._defaults["metatarsal"]["length_mm"])
        self._meta_length.valueChanged.connect(self._update_meta_preview)

        self._meta_width = QDoubleSpinBox()
        self._meta_width.setRange(0.5, 6.0)
        self._meta_width.setSingleStep(0.5)
        self._meta_width.setValue(self._defaults["metatarsal"]["width_mm"])
        self._meta_width.valueChanged.connect(self._update_meta_preview)

        self._meta_angle = QDoubleSpinBox()
        self._meta_angle.setRange(-90.0, 90.0)
        self._meta_angle.setSingleStep(5.0)
        self._meta_angle.setValue(self._defaults["metatarsal"]["angle_deg"])
        self._meta_angle.valueChanged.connect(self._update_meta_preview)

        self._meta_smooth = QDoubleSpinBox()
        self._meta_smooth.setRange(0.0, 3.0)
        self._meta_smooth.setSingleStep(0.5)
        self._meta_smooth.setValue(self._defaults["metatarsal"]["smoothing"])
        self._meta_smooth.valueChanged.connect(self._update_meta_preview)

        self._meta_front_offset = QDoubleSpinBox()
        self._meta_front_offset.setRange(0.00, 0.40)
        self._meta_front_offset.setSingleStep(0.05)
        self._meta_front_offset.setValue(self._defaults["metatarsal"]["front_offset"])
        self._meta_front_offset.setDecimals(2)
        self._meta_front_offset.valueChanged.connect(self._update_meta_preview)

        form.addRow("高さ (mm):", self._meta_height)
        form.addRow("長さ (cm):", self._meta_length)
        form.addRow("幅 (cm):", self._meta_width)
        form.addRow("角度 (°):", self._meta_angle)
        form.addRow("スムージング:", self._meta_smooth)
        form.addRow("前方オフセット:", self._meta_front_offset)
        layout.addLayout(form)

        self._meta_pos_label = QLabel("位置: 未設定")
        layout.addWidget(self._meta_pos_label)

        self._meta_preview_stats_label = QLabel("")
        self._meta_preview_stats_label.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        layout.addWidget(self._meta_preview_stats_label)

        btn_apply = QPushButton("② メタターサルサポートを適用")
        btn_apply.clicked.connect(self._apply_metatarsal)
        layout.addWidget(btn_apply)

        self._meta_mirror_check = QCheckBox("反対足にも適用（ミラー）")
        layout.addWidget(self._meta_mirror_check)

        self._meta_apply_stats_label = QLabel("")
        self._meta_apply_stats_label.setWordWrap(True)
        self._meta_apply_stats_label.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        layout.addWidget(self._meta_apply_stats_label)

        btn_reset_meta = QPushButton("設定値を初期化")
        btn_reset_meta.clicked.connect(self._reset_meta_params)
        layout.addWidget(btn_reset_meta)

        step_row2 = QHBoxLayout()
        btn_undo2 = QPushButton("↩ 戻す")
        btn_undo2.clicked.connect(self._undo)
        btn_redo2 = QPushButton("↪ 進む")
        btn_redo2.clicked.connect(self._redo)
        step_row2.addWidget(btn_undo2)
        step_row2.addWidget(btn_redo2)
        layout.addLayout(step_row2)

        layout.addStretch()
        return w

    # ─── 患者情報タブ ───
    def _build_patient_tab(self) -> QWidget:
        w = QWidget()
        layout = QFormLayout(w)
        layout.setSpacing(6)

        self._pt_name = QLineEdit()
        self._pt_age = QLineEdit()
        self._pt_gender = QComboBox()
        self._pt_gender.addItems(["Male", "Female"])
        self._pt_shoe_eur = QLineEdit()
        self._pt_shoe_jp = QLineEdit()

        layout.addRow("氏名:", self._pt_name)
        layout.addRow("年齢:", self._pt_age)
        layout.addRow("性別:", self._pt_gender)
        layout.addRow("サイズ (EUR):", self._pt_shoe_eur)
        layout.addRow("サイズ (JP cm):", self._pt_shoe_jp)

        btn_save_pt = QPushButton("患者情報を保存")
        btn_save_pt.clicked.connect(self._save_patient_info)
        layout.addRow(btn_save_pt)

        return w

    # ──────────────────────────────────────────
    # ファイル操作
    # ──────────────────────────────────────────
    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "GRDファイルを開く", "", "GRD Files (*.grd);;All Files (*)"
        )
        if not path:
            return
        try:
            grd = grd_io.load(path)
            self._model = FootModel(grd)
            self._current_path = path
            self._refresh_heatmaps()
            self._load_patient_fields()
            self._update_status(f"読み込み完了: {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"ファイル読み込み失敗:\n{e}")

    def _save_file(self):
        if not self._model:
            return
        try:
            grd_io.save(self._model.grd, self._current_path)
            self._update_status(f"保存完了: {Path(self._current_path).name}")
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"保存失敗:\n{e}")

    def _save_as_file(self):
        if not self._model:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "名前をつけて保存", "", "GRD Files (*.grd);;All Files (*)"
        )
        if not path:
            return
        try:
            grd_io.save(self._model.grd, path)
            self._current_path = path
            self._update_status(f"保存完了: {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"保存失敗:\n{e}")

    # ──────────────────────────────────────────
    # 編集操作
    # ──────────────────────────────────────────
    def _undo(self):
        if self._model and self._model.undo():
            self._refresh_heatmaps()
            self._update_status("元に戻しました")

    def _redo(self):
        if self._model and self._model.redo():
            self._refresh_heatmaps()
            self._update_status("やり直しました")

    def _reset(self):
        if self._model:
            reply = QMessageBox.question(
                self, "確認", "すべての変更をリセットしますか？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._model.reset()
                self._refresh_heatmaps()
                self._update_status("リセットしました")

    # ──────────────────────────────────────────
    # ノイズ除去
    # ──────────────────────────────────────────
    def _run_noise_removal(self):
        if not self._model:
            QMessageBox.information(self, "情報", "先にGRDファイルを開いてください")
            return
        n = self._model.remove_noise()
        self._refresh_heatmaps()
        self._update_status(f"ノイズ除去: {n} ピンをゼロ化しました")

    def _toggle_erase_select(self, checked: bool):
        self._erase_select_mode = checked
        self._btn_erase_select.setText(
            "選択モード ON (ドラッグして選択)" if checked else "選択モード OFF"
        )
        foot = "left" if self._erase_foot_combo.currentIndex() == 0 else "right"
        self._hm_left.set_select_mode(checked and foot == "left")
        self._hm_right.set_select_mode(checked and foot == "right")
        if checked and self._btn_arch_select.isChecked():
            self._btn_arch_select.setChecked(False)

    def _clear_erase_selection(self):
        self._hm_left.clear_selection()
        self._hm_right.clear_selection()

    def _apply_erase(self):
        if not self._model:
            QMessageBox.information(self, "情報", "先にGRDファイルを開いてください")
            return
        foot = "left" if self._erase_foot_combo.currentIndex() == 0 else "right"
        hm = self._hm_left if foot == "left" else self._hm_right
        mask = hm.get_selection_mask()
        if not mask.any():
            QMessageBox.information(self, "情報", "消去する領域をドラッグして選択してください")
            return
        n = self._model.erase_cells(foot, mask)
        self._btn_erase_select.setChecked(False)
        hm.clear_selection()
        self._refresh_heatmaps()
        self._update_status(f"手動消去: {mask.sum()} セル選択 / {n} ピンをゼロ化")

    def _apply_mirror_mask(self):
        if not self._model:
            QMessageBox.information(self, "情報", "先にGRDファイルを開いてください")
            return
        reference_foot = "left" if self._mirror_ref_combo.currentIndex() == 0 else "right"
        margin = self._mirror_margin.value()
        n = self._model.apply_mirror_mask(reference_foot, margin)
        if n == 0 and not (self._model.grd.grid[
                grd_io.LEFT_ROWS if reference_foot == "left" else grd_io.RIGHT_ROWS
            ] < 0).any():
            QMessageBox.information(self, "情報", "参照足にデータがありません")
            return
        self._refresh_heatmaps()
        target = "右足" if reference_foot == "left" else "左足"
        self._update_status(f"対称マスク: {target}から {n} ピンをゼロ化 (余白{margin:.1f}cm)")

    def _apply_mirror_copy(self):
        if not self._model:
            QMessageBox.information(self, "情報", "先にGRDファイルを開いてください")
            return
        source = "left" if self._mirror_copy_combo.currentIndex() == 0 else "right"
        src_label = "左足" if source == "left" else "右足"
        tgt_label = "右足" if source == "left" else "左足"
        reply = QMessageBox.question(
            self, "確認",
            f"{src_label}のデータを左右反転して{tgt_label}に上書きしますか？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._model.mirror_foot(source)
            self._refresh_heatmaps()
            self._update_status(f"ミラーコピー: {src_label} → {tgt_label}")

    def _on_tab_changed(self, index: int):
        if index != 2:
            self._hm_left.set_overlay(None)
            self._hm_right.set_overlay(None)
            self._meta_preview_stats_label.setText("")
        if index != 1:
            self._arch_preview_label.setText("")
            self._hm_left.set_mirror_mask(None)
            self._hm_right.set_mirror_mask(None)

    # ──────────────────────────────────────────
    # アーチ調整
    # ──────────────────────────────────────────
    def _toggle_arch_select(self, checked: bool):
        self._btn_arch_select.setText(
            "領域選択モード ON (ドラッグして選択)" if checked else "領域選択モード OFF"
        )
        foot = "left" if self._arch_foot_combo.currentIndex() == 0 else "right"
        self._hm_left.set_select_mode(checked and foot == "left")
        self._hm_right.set_select_mode(checked and foot == "right")
        if not checked:
            self._hm_left.set_mirror_mask(None)
            self._hm_right.set_mirror_mask(None)
        if checked and self._btn_erase_select.isChecked():
            self._btn_erase_select.setChecked(False)

    def _clear_arch_selection(self):
        self._hm_left.clear_selection()
        self._hm_right.clear_selection()
        self._hm_left.set_mirror_mask(None)
        self._hm_right.set_mirror_mask(None)

    def _on_arch_mirror_toggled(self, checked: bool):
        if not checked:
            self._hm_left.set_mirror_mask(None)
            self._hm_right.set_mirror_mask(None)
        else:
            foot = "left" if self._arch_foot_combo.currentIndex() == 0 else "right"
            hm = self._hm_left if foot == "left" else self._hm_right
            mask = hm.get_selection_mask()
            if mask.any():
                other = self._hm_right if foot == "left" else self._hm_left
                other.set_mirror_mask(np.fliplr(mask))

    def _on_arch_slider(self, val: int):
        self._arch_params.height_mm = val / 2.0
        self._update_arch_preview_label()

    def _on_selection_changed(self, foot: str, mask: np.ndarray):
        if self._erase_select_mode:
            pass  # erase適用時にget_selection_maskで取得するため保持不要
        else:
            self._arch_params.mask = mask
            other = self._hm_right if foot == "left" else self._hm_left
            if self._tabs.currentIndex() == 1 and self._arch_mirror_check.isChecked():
                other.set_mirror_mask(np.fliplr(mask))
            else:
                other.set_mirror_mask(None)
        self._update_arch_preview_label()

    def _update_arch_preview_label(self):
        mask = self._arch_params.mask
        if mask is None or not mask.any():
            self._arch_preview_label.setText("")
            return
        actual = preview_arch_max(mask, self._arch_slider.value() / 2.0, self._arch_smooth.value())
        self._arch_preview_label.setText(f"プレビュー最大: {actual:.1f}mm")

    def _reset_arch_params(self):
        self._arch_slider.setValue(int(self._defaults["arch"]["height_mm"] * 2))
        self._arch_smooth.setValue(self._defaults["arch"]["smoothing"])

    def _reset_meta_params(self):
        self._meta_height.setValue(self._defaults["metatarsal"]["height_mm"])
        self._meta_length.setValue(self._defaults["metatarsal"]["length_mm"])
        self._meta_width.setValue(self._defaults["metatarsal"]["width_mm"])
        self._meta_angle.setValue(self._defaults["metatarsal"]["angle_deg"])
        self._meta_smooth.setValue(self._defaults["metatarsal"]["smoothing"])
        self._meta_front_offset.setValue(self._defaults["metatarsal"]["front_offset"])

    def _open_settings(self):
        dlg = SettingsDialog(self._defaults, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            values = dlg.get_values()
            _save_settings(values)
            self._defaults = values

    def _apply_arch(self):
        if not self._model:
            QMessageBox.information(self, "情報", "先にGRDファイルを開いてください")
            return
        foot = "left" if self._arch_foot_combo.currentIndex() == 0 else "right"
        hm = self._hm_left if foot == "left" else self._hm_right
        mask = hm.get_selection_mask()
        if not mask.any():
            QMessageBox.information(self, "情報", "アーチ領域をドラッグして選択してください")
            return
        params = ArchParams(
            mask=mask,
            height_mm=self._arch_slider.value() / 2.0,
            smoothing=self._arch_smooth.value(),
        )
        stats = self._model.apply_arch(foot, params)
        self._btn_arch_select.setChecked(False)
        hm.clear_selection()
        self._hm_left.set_mirror_mask(None)
        self._hm_right.set_mirror_mask(None)
        if self._arch_mirror_check.isChecked():
            other_foot = "right" if foot == "left" else "left"
            mirror_params = ArchParams(
                mask=np.fliplr(mask),
                height_mm=params.height_mm,
                smoothing=params.smoothing,
            )
            self._model.apply_arch(other_foot, mirror_params)
        self._refresh_heatmaps()
        if self._arch_mirror_check.isChecked():
            self._update_status(f"アーチ調整を両足に適用 ({params.height_mm:.1f}mm)")
        else:
            self._update_status(f"アーチ調整を適用 ({foot}, {params.height_mm:.1f}mm)")
        if stats:
            self._arch_stats_label.setText(
                f"設定: {stats['set_mm']:.1f}mm ／ 実測最大: {stats['actual_max']:.1f}mm ／ 影響: {stats['affected']}セル"
            )

    # ──────────────────────────────────────────
    # メタターサルサポート
    # ──────────────────────────────────────────
    def _on_cell_clicked(self, foot: str, row: int, col: int):
        # メタターサルタブのときのみ配置・プレビュー更新
        if self._tabs.currentIndex() != 2:
            return
        self._meta_params.center_row = row
        self._meta_params.center_col = col
        self._meta_pos_label.setText(f"位置: 行{row+1} 列{col+1}")
        hm = self._hm_left if foot == "left" else self._hm_right
        hm.set_meta_center((row, col))
        other = self._hm_right if foot == "left" else self._hm_left
        if self._meta_mirror_check.isChecked():
            other.set_meta_center((row, 15 - col))
        else:
            other.set_meta_center(None)
        self._update_meta_preview()

    def _update_meta_preview(self):
        if not self._model:
            return
        foot = "left" if self._meta_foot_combo.currentIndex() == 0 else "right"
        grid = self._model.left_grid if foot == "left" else self._model.right_grid
        from foot_model import _make_teardrop_bump
        common_args = dict(
            shape=grid.shape,
            center_rc=(self._meta_params.center_row, self._meta_params.center_col),
            height=self._meta_height.value(),
            length=self._meta_length.value(),
            width=self._meta_width.value(),
            angle_deg=self._meta_angle.value(),
            front_offset=self._meta_front_offset.value(),
        )
        bump = _make_teardrop_bump(**common_args, smoothing=self._meta_smooth.value())
        boundary_mask = _make_teardrop_bump(**common_args, smoothing=0) > 0.01
        hm = self._hm_left if foot == "left" else self._hm_right
        hm.set_overlay(bump, boundary_mask)
        other = self._hm_right if foot == "left" else self._hm_left
        if self._meta_mirror_check.isChecked():
            mirror_col = 15.0 - self._meta_params.center_col
            mirror_bump = _make_teardrop_bump(
                shape=grid.shape,
                center_rc=(self._meta_params.center_row, mirror_col),
                height=self._meta_height.value(),
                length=self._meta_length.value(),
                width=self._meta_width.value(),
                angle_deg=-self._meta_angle.value(),
                front_offset=self._meta_front_offset.value(),
                smoothing=self._meta_smooth.value(),
            )
            mirror_boundary = _make_teardrop_bump(
                shape=grid.shape,
                center_rc=(self._meta_params.center_row, mirror_col),
                height=self._meta_height.value(),
                length=self._meta_length.value(),
                width=self._meta_width.value(),
                angle_deg=-self._meta_angle.value(),
                front_offset=self._meta_front_offset.value(),
                smoothing=0,
            ) > 0.01
            other.set_overlay(mirror_bump, mirror_boundary)
        else:
            other.set_overlay(None)
        peak = float(np.max(bump)) if bump is not None else 0.0
        self._meta_preview_stats_label.setText(f"プレビュー最大: {peak:.1f}mm")

    def _apply_metatarsal(self):
        if not self._model:
            QMessageBox.information(self, "情報", "先にGRDファイルを開いてください")
            return
        foot = "left" if self._meta_foot_combo.currentIndex() == 0 else "right"
        params = MetatarsalParams(
            center_row=self._meta_params.center_row,
            center_col=self._meta_params.center_col,
            height_mm=self._meta_height.value(),
            length_mm=self._meta_length.value(),
            width_mm=self._meta_width.value(),
            angle_deg=self._meta_angle.value(),
            smoothing=self._meta_smooth.value(),
            front_offset=self._meta_front_offset.value(),
        )
        stats = self._model.apply_metatarsal(foot, params)
        hm = self._hm_left if foot == "left" else self._hm_right
        hm.set_overlay(None)
        hm.set_meta_center(None)
        other = self._hm_right if foot == "left" else self._hm_left
        other.set_overlay(None)
        other.set_meta_center(None)
        if self._meta_mirror_check.isChecked():
            other_foot = "right" if foot == "left" else "left"
            mirror_params = MetatarsalParams(
                center_row=params.center_row,
                center_col=15.0 - params.center_col,
                height_mm=params.height_mm,
                length_mm=params.length_mm,
                width_mm=params.width_mm,
                angle_deg=-params.angle_deg,
                smoothing=params.smoothing,
                front_offset=params.front_offset,
            )
            self._model.apply_metatarsal(other_foot, mirror_params)
        self._refresh_heatmaps()
        if self._meta_mirror_check.isChecked():
            self._update_status(f"メタターサルサポートを両足に適用 ({params.height_mm:.1f}mm)")
        else:
            self._update_status(f"メタターサルサポートを適用 ({foot}, {params.height_mm:.1f}mm)")
        self._meta_apply_stats_label.setText(
            f"設定: {stats['set_mm']:.1f}mm ／ 実測最大: {stats['actual_max']:.1f}mm ／ 影響: {stats['affected']}セル"
        )
        self._meta_preview_stats_label.setText("")

    # ──────────────────────────────────────────
    # 患者情報
    # ──────────────────────────────────────────
    def _load_patient_fields(self):
        if not self._model:
            return
        p = self._model.grd.patient
        self._pt_name.setText(p.name)
        self._pt_age.setText(p.age)
        idx = 1 if p.gender.lower() == "female" else 0
        self._pt_gender.setCurrentIndex(idx)
        self._pt_shoe_eur.setText(p.shoe_size_eur)
        self._pt_shoe_jp.setText(p.shoe_size_jp)

    def _save_patient_info(self):
        if not self._model:
            return
        p = self._model.grd.patient
        p.name = self._pt_name.text()
        p.age = self._pt_age.text()
        p.gender = self._pt_gender.currentText()
        p.shoe_size_eur = self._pt_shoe_eur.text()
        p.shoe_size_jp = self._pt_shoe_jp.text()
        # raw_meta_linesを更新
        _update_raw_meta(self._model.grd)
        self._update_status("患者情報を更新しました")

    # ──────────────────────────────────────────
    # 3D表示
    # ──────────────────────────────────────────
    def _show_3d_view(self):
        if not self._model:
            QMessageBox.information(self, "情報", "先にGRDファイルを開いてください")
            return

        if self._3d_dialog is not None:
            self._3d_dialog.raise_()
            self._3d_dialog.activateWindow()
            return

        import matplotlib
        matplotlib.rcParams['font.family'] = ['MS Gothic', 'Meiryo', 'Yu Gothic', 'sans-serif']

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
        from PyQt6.QtCore import Qt

        dlg = QDialog(self)
        dlg.setWindowTitle("足底3D表示")
        dlg.resize(1100, 780)
        dlg.setStyleSheet("background-color: #111111;")

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.setSpacing(4)

        fig = Figure(figsize=(13, 7), facecolor='black')
        fig.suptitle("足底3D表示", fontsize=13, color='white')
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        feet = [
            (self._model.left_grid, "左足"),
            (self._model.right_grid, "右足"),
        ]
        self._3d_axes = []
        self._3d_wireframes = []
        self._3d_base_wireframes: list = [None, None]
        base_grids = [self._model.base_left_grid, self._model.base_right_grid]
        for i, (grid, label) in enumerate(feet):
            ax = fig.add_subplot(1, 2, i + 1, projection='3d')
            ax.set_facecolor('black')
            rows, cols = grid.shape
            X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
            Z = np.flipud(grid) / 10

            # 元データサーフェスを先に描画（差分表示ONのとき）
            if self._btn_diff.isChecked():
                Z_base = np.flipud(base_grids[i]) / 10
                bwf = ax.plot_surface(X, Y, Z_base, color='#666666',
                                      rstride=1, cstride=1, alpha=1.0, edgecolor='none')
                self._3d_base_wireframes[i] = bwf

            # 修正後データの描画（差分ON=カラーマップサーフェス / OFF=青ワイヤーフレーム）
            if self._btn_diff.isChecked():
                import matplotlib.colors, matplotlib.cm
                Z_diff = Z - Z_base
                vmax = max(float(np.max(Z_diff)), 0.1)
                norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
                face_colors = matplotlib.cm.plasma(norm(Z_diff))
                wf = ax.plot_surface(X, Y, Z, facecolors=face_colors,
                                     rstride=1, cstride=1, alpha=0.95, edgecolor='none')
            else:
                wf = ax.plot_wireframe(X, Y, Z, color='#2266ee', linewidth=0.5,
                                       rstride=1, cstride=1, alpha=0.9)

            ax.set_title(label, fontsize=12, color='white', pad=-15)
            ax.set_box_aspect([cols, rows, 2])
            ax.view_init(elev=25, azim=-55)
            ax.set_axis_off()
            self._3d_axes.append(ax)
            self._3d_wireframes.append(wf)

        self._3d_canvas = canvas
        self._3d_x_flip = [False] * len(self._3d_axes)
        self._3d_y_flip = [False] * len(self._3d_axes)

        fig.tight_layout()
        canvas.draw()

        axes = self._3d_axes
        # 軸反転状態のトラッキング（invert_xaxis/yaxisはトグルのため）
        x_flip_state = self._3d_x_flip
        y_flip_state = self._3d_y_flip

        # 視点切替ボタン（PyQt6ネイティブ）
        # タプル: (ラベル, elev, azim, flip_x, flip_y)
        # flip_x: 後方視点でX軸（足幅方向）が自然に反転するため補正
        # flip_y: 矢状面で内側(azim=-145)から見るとY軸（つま先↔踵）が
        #         外側(azim=-35)と逆になるため補正
        # タプル: (ラベル, elev, azim, flip_x, flip_y_left, flip_y_right)
        # 矢状面は左右足で内側/外側が解剖学的に鏡像なため flip_y を足ごとに分ける
        view_groups = [
            ("前額面", [
                ("前方", 0, -90, False, False, False),
                ("後方", 0, 90, True, False, False),
            ]),
            ("矢状面", [
                ("内側", 0, -145, False, True, False),
                ("外側", 0, -35, False, False, True),
            ]),
            ("水平面", [
                ("上側", 90, -90, False, False, False),
                ("下側", -80, -90, False, False, False),
            ]),
        ]

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(20, 0, 20, 0)
        btn_row.addStretch()

        btn_style = (
            "QPushButton { background-color: #404040; color: white; "
            "border-radius: 4px; font-size: 12px; padding: 4px 16px; }"
            "QPushButton:hover { background-color: #2255aa; }"
            "QPushButton:pressed { background-color: #113388; }"
        )
        lbl_style = "color: #aaaaaa; font-size: 10px; background: transparent;"

        for group_name, views in view_groups:
            group_col = QVBoxLayout()
            group_col.setSpacing(3)

            group_lbl = QLabel(group_name)
            group_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            group_lbl.setStyleSheet(lbl_style)
            group_col.addWidget(group_lbl)

            pair_row = QHBoxLayout()
            pair_row.setSpacing(6)
            for view_label, elev, azim, flip_x, flip_y_left, flip_y_right in views:
                btn = QPushButton(view_label)
                btn.setFixedHeight(30)
                btn.setStyleSheet(btn_style)

                def make_cb(e, a, fx, fy_l, fy_r):
                    def cb():
                        for i, ax in enumerate(axes):
                            ax.view_init(elev=e, azim=a)
                            if x_flip_state[i] != fx:
                                ax.invert_xaxis()
                                x_flip_state[i] = fx
                            fy = fy_r if i == 1 else fy_l
                            if y_flip_state[i] != fy:
                                ax.invert_yaxis()
                                y_flip_state[i] = fy
                        canvas.draw()
                    return cb

                btn.clicked.connect(make_cb(elev, azim, flip_x, flip_y_left, flip_y_right))
                pair_row.addWidget(btn)

            group_col.addLayout(pair_row)
            btn_row.addLayout(group_col)
            btn_row.addStretch()

        layout.addLayout(btn_row)

        def _on_3d_closed():
            self._3d_dialog = None
            self._3d_canvas = None
            self._3d_axes = []
            self._3d_wireframes = []

        dlg.finished.connect(_on_3d_closed)
        self._3d_dialog = dlg
        dlg.show()

    # ──────────────────────────────────────────
    # ユーティリティ
    # ──────────────────────────────────────────
    def _refresh_heatmaps(self):
        if not self._model:
            return
        self._hm_left.set_grid(self._model.left_grid)
        self._hm_right.set_grid(self._model.right_grid)
        self._refresh_layer_list()
        self._refresh_3d()
        self._update_diff_overlay()

    def _refresh_layer_list(self):
        if not self._model:
            self._layer_list.clear()
            return
        self._layer_list.blockSignals(True)
        self._layer_list.clear()
        for i, layer in enumerate(self._model.layers):
            item = QListWidgetItem(layer.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if layer.enabled else Qt.CheckState.Unchecked
            )
            item.setData(Qt.ItemDataRole.UserRole, i)
            self._layer_list.addItem(item)
        self._layer_list.scrollToBottom()
        self._layer_list.blockSignals(False)

    def _on_layer_item_changed(self, item: QListWidgetItem):
        if not self._model:
            return
        i = item.data(Qt.ItemDataRole.UserRole)
        enabled = item.checkState() == Qt.CheckState.Checked
        if self._model.layers[i].enabled != enabled:
            self._model.toggle_layer(i)
            self._hm_left.set_grid(self._model.left_grid)
            self._hm_right.set_grid(self._model.right_grid)
            self._refresh_3d()
            name = self._model.layers[i].name
            state = "有効" if enabled else "無効"
            self._update_status(f"レイヤー {state}: {name}")

    def _on_toggle_all_layers(self, checked: bool) -> None:
        if not self._model:
            self._btn_toggle_all.setChecked(False)
            return
        self._model.set_all_enabled(not checked)
        self._btn_toggle_all.setText("補正 全ON に戻す" if checked else "補正 全OFF")
        self._refresh_heatmaps()
        msg = "全補正を一時OFFにしました" if checked else "全補正を再有効化しました"
        self._update_status(msg)

    def _on_diff_toggled(self, checked: bool) -> None:
        if not checked:
            self._hm_left.set_diff_grid(None)
            self._hm_right.set_diff_grid(None)
        else:
            self._update_diff_overlay()
        self._refresh_3d()

    def _update_diff_overlay(self) -> None:
        if not self._model or not self._btn_diff.isChecked():
            return
        self._hm_left.set_diff_grid(self._model.left_grid - self._model.base_left_grid)
        self._hm_right.set_diff_grid(self._model.right_grid - self._model.base_right_grid)

    def _refresh_3d(self):
        if not self._3d_canvas or not self._3d_axes or not self._model:
            return
        feet = [
            (self._model.left_grid, "左足"),
            (self._model.right_grid, "右足"),
        ]
        for i, (ax, (grid, _label)) in enumerate(zip(self._3d_axes, feet)):
            # 視点・軸範囲を保存
            elev, azim = ax.elev, ax.azim
            xlim = ax.get_xlim3d()
            ylim = ax.get_ylim3d()
            zlim = ax.get_zlim3d()
            rows, cols = grid.shape
            X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
            Z = np.flipud(grid) / 10
            # 元データサーフェスを先に削除→再描画（差分表示ONのとき）
            bwf = self._3d_base_wireframes[i] if i < len(self._3d_base_wireframes) else None
            if bwf is not None:
                try:
                    bwf.remove()
                except (ValueError, AttributeError):
                    pass
                self._3d_base_wireframes[i] = None
            if hasattr(self, '_btn_diff') and self._btn_diff.isChecked():
                base_grids = [self._model.base_left_grid, self._model.base_right_grid]
                Z_base = np.flipud(base_grids[i]) / 10
                bwf = ax.plot_surface(X, Y, Z_base, color='#666666',
                                      rstride=1, cstride=1, alpha=1.0, edgecolor='none')
                self._3d_base_wireframes[i] = bwf
            # 修正後データの差し替え（差分ON=カラーマップサーフェス / OFF=青ワイヤーフレーム）
            try:
                self._3d_wireframes[i].remove()
            except (ValueError, IndexError):
                pass
            if hasattr(self, '_btn_diff') and self._btn_diff.isChecked():
                import matplotlib.colors, matplotlib.cm
                Z_diff = Z - Z_base
                vmax = max(float(np.max(Z_diff)), 0.1)
                norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
                face_colors = matplotlib.cm.plasma(norm(Z_diff))
                wf = ax.plot_surface(X, Y, Z, facecolors=face_colors,
                                     rstride=1, cstride=1, alpha=0.95, edgecolor='none')
            else:
                wf = ax.plot_wireframe(X, Y, Z, color='#2266ee', linewidth=0.5,
                                       rstride=1, cstride=1, alpha=0.9)
            self._3d_wireframes[i] = wf
            # autoscalingで変化した軸範囲・視点を復元
            ax.set_xlim3d(xlim)
            ax.set_ylim3d(ylim)
            ax.set_zlim3d(zlim)
            ax.view_init(elev=elev, azim=azim)
        self._3d_canvas.draw()

    def _update_status(self, msg: str):
        self.statusBar().showMessage(msg)


def _update_raw_meta(grd: grd_io.GrdData):
    """患者情報をraw_meta_linesに書き戻す（固定位置ベース）"""
    p = grd.patient
    lines = grd.raw_meta_lines
    def set_line(idx: int, val: str):
        while len(lines) <= idx:
            lines.append("")
        lines[idx] = val

    set_line(0, f"_({p.name})")
    set_line(5, p.country)
    set_line(11, p.age)
    set_line(12, p.gender_code)
    set_line(13, p.gender)
    set_line(17, p.device_id)
    set_line(20, p.shoe_size_eur)
    set_line(21, p.shoe_size_jp)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
