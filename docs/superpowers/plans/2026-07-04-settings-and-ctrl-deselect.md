# 設定保存 & Ctrl+クリック選択解除 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** アーチ範囲のCtrl+クリックセル単位トグルと、アーチ・メタターサル初期値のJSON保存・設定ダイアログを追加する。

**Architecture:** `heatmap_widget.py` の `mousePressEvent` にCtrlキー判定を追加。`main.py` に `SettingsDialog` クラスとメニューバーを追加し、`platformdirs` で取得したユーザー設定フォルダに `settings.json` を読み書きする。

**Tech Stack:** PyQt6, platformdirs, json, pathlib

## Global Constraints

- PyQt6 のみ（PySide6不可）
- `any` 型注釈は使わず `unknown` 相当の厳密な型を使う（Python: `dict[str, Any]` ではなく具体的な型 or TypedDict）
- コミットは Conventional Commits 形式、本文は日本語
- テストコードは確認なしに削除・コメントアウトしない
- README・ドキュメントは変更しない

---

### Task 1: Ctrl+クリックでセル単位トグル

**Files:**
- Modify: `riki_footscan/heatmap_widget.py:270-279`

**Interfaces:**
- Consumes: 既存の `mousePressEvent`、`_sel_mask: np.ndarray`、`_select_mode: bool`
- Produces: 変更なし（シグナルは既存の `selectionChanged(object)` を流用）

- [ ] **Step 1: `mousePressEvent` を修正**

`heatmap_widget.py` の `mousePressEvent`（line 270）を以下に置き換える：

```python
def mousePressEvent(self, event):
    if event.button() == Qt.MouseButton.LeftButton:
        r, c = self._px_to_grid(event.position().x(), event.position().y())
        if 0 <= r < 32 and 0 <= c < 16:
            ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            self.cellClicked.emit(r, c)
            if self._select_mode:
                if ctrl:
                    self._sel_mask[r, c] = not self._sel_mask[r, c]
                    self.selectionChanged.emit(self._sel_mask.copy())
                    self.update()
                else:
                    self._is_selecting = True
                    self._sel_mask[r, c] = True
                    self.selectionChanged.emit(self._sel_mask.copy())
                    self.update()
```

- [ ] **Step 2: 手動動作確認**

```
python main.py
```
1. GRDファイルを開く
2. アーチタブ → 「領域選択モード ON」
3. ヒートマップをドラッグで複数セル選択
4. Ctrl+クリックで選択済みセルが解除されること（ハイライトが消える）を確認
5. Ctrl+クリックで未選択セルが選択されること（ハイライトが付く）を確認
6. 通常クリック（Ctrlなし）は従来通り追加選択のままであることを確認

- [ ] **Step 3: コミット**

```bash
git add riki_footscan/heatmap_widget.py
git commit -m "feat: Ctrl+クリックでアーチ選択セルをトグル解除できるよう追加"
```

---

### Task 2: 設定読み書きユーティリティ

**Files:**
- Modify: `riki_footscan/main.py` — `import` 追加、モジュールレベル関数2つ追加

**Interfaces:**
- Produces:
  - `_DEFAULT_SETTINGS: dict` — フォールバック用デフォルト値定数
  - `_load_settings() -> dict` — JSONから読み込み（失敗時はデフォルト）
  - `_save_settings(values: dict) -> None` — JSONへ書き込み

- [ ] **Step 1: `import` を追加**

`main.py` の既存 import ブロック末尾（`from heatmap_widget import HeatmapWidget` の後）に追加：

```python
import json
import copy
from platformdirs import user_config_dir
```

- [ ] **Step 2: デフォルト定数と関数を追加**

`main.py` の `class MainWindow` 定義の直前（`class MainWindow(QMainWindow):` の1行上）に追加：

```python
_DEFAULT_SETTINGS: dict = {
    "arch": {"height_mm": 5, "smoothing": 1.5},
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
```

- [ ] **Step 3: 構文エラーがないか確認**

```bash
python -c "import ast; ast.parse(open('riki_footscan/main.py').read()); print('OK')"
```

Expected: `OK`

---

### Task 3: `SettingsDialog` クラス追加

**Files:**
- Modify: `riki_footscan/main.py` — `SettingsDialog` クラスを `MainWindow` の直前に追加

**Interfaces:**
- Consumes: `_DEFAULT_SETTINGS`
- Produces: `SettingsDialog(defaults: dict, parent=None)` — `exec()` 後に `get_values() -> dict` で値を取得

- [ ] **Step 1: `QDialog` を import に追加**

`main.py` の `from PyQt6.QtWidgets import (...)` ブロックに `QDialog` を追加：

```python
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGroupBox, QPushButton, QLabel, QSlider, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QTabWidget, QScrollArea, QLineEdit,
    QComboBox, QCheckBox, QSplitter, QStatusBar, QFormLayout,
    QListWidget, QListWidgetItem, QDialog,
)
```

- [ ] **Step 2: `SettingsDialog` クラスを追加**

`_load_settings` 関数の直後、`class MainWindow` の直前に追加：

```python
class SettingsDialog(QDialog):
    def __init__(self, defaults: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("設定")
        self.setMinimumWidth(340)
        layout = QVBoxLayout(self)

        # ── アーチ初期値 ──
        arch_box = QGroupBox("アーチ調整 初期値")
        arch_form = QFormLayout(arch_box)

        self._arch_height = QSpinBox()
        self._arch_height.setRange(1, 20)
        self._arch_height.setSuffix(" mm")
        self._arch_height.setValue(int(defaults["arch"]["height_mm"]))
        arch_form.addRow("調整量:", self._arch_height)

        self._arch_smooth = QDoubleSpinBox()
        self._arch_smooth.setRange(0.5, 5.0)
        self._arch_smooth.setSingleStep(0.5)
        self._arch_smooth.setSuffix(" cm")
        self._arch_smooth.setValue(defaults["arch"]["smoothing"])
        arch_form.addRow("スムージング幅:", self._arch_smooth)

        layout.addWidget(arch_box)

        # ── メタターサル初期値 ──
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

        # ── ボタン ──
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
```

- [ ] **Step 3: 構文チェック**

```bash
python -c "import ast; ast.parse(open('riki_footscan/main.py').read()); print('OK')"
```

Expected: `OK`

---

### Task 4: MainWindow への設定統合

**Files:**
- Modify: `riki_footscan/main.py` — `__init__`、`_build_ui`、`_build_arch_tab`、`_build_meta_tab`、`_reset_arch_params`、`_reset_meta_params` を修正、`_open_settings` を追加

**Interfaces:**
- Consumes: `_load_settings()`, `_save_settings()`, `SettingsDialog`
- Produces: `self._defaults: dict` (インスタンス変数として保持)

- [ ] **Step 1: `__init__` に設定読み込みを追加**

`MainWindow.__init__` の `self._build_ui()` 呼び出しの直前に追加：

```python
self._defaults = _load_settings()
```

つまり `__init__` は以下の順序になる：

```python
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

    self._defaults = _load_settings()  # ← 追加
    self._build_ui()
    self._update_status("GRDファイルを開いてください")
```

- [ ] **Step 2: `_build_ui` にメニューバーを追加**

`_build_ui` の先頭（`central = QWidget()` の前）に追加：

```python
menu_bar = self.menuBar()
settings_menu = menu_bar.addMenu("設定")
act_settings = settings_menu.addAction("設定を開く...")
act_settings.triggered.connect(self._open_settings)
```

- [ ] **Step 3: `_build_arch_tab` の初期値をデフォルト設定から取得**

`_build_arch_tab` の該当行を変更：

```python
# 変更前
self._arch_slider.setValue(5)
self._arch_spin.setValue(5)
self._arch_smooth.setValue(1.5)

# 変更後
self._arch_slider.setValue(self._defaults["arch"]["height_mm"])
self._arch_spin.setValue(self._defaults["arch"]["height_mm"])
self._arch_smooth.setValue(self._defaults["arch"]["smoothing"])
```

- [ ] **Step 4: `_build_meta_tab` の初期値をデフォルト設定から取得**

`_build_meta_tab` の該当行を変更：

```python
# 変更前
self._meta_height.setValue(5.0)
self._meta_length.setValue(4.0)
self._meta_width.setValue(2.5)
self._meta_angle.setValue(0.0)
self._meta_smooth.setValue(1.0)
self._meta_front_offset.setValue(0.00)

# 変更後
self._meta_height.setValue(self._defaults["metatarsal"]["height_mm"])
self._meta_length.setValue(self._defaults["metatarsal"]["length_mm"])
self._meta_width.setValue(self._defaults["metatarsal"]["width_mm"])
self._meta_angle.setValue(self._defaults["metatarsal"]["angle_deg"])
self._meta_smooth.setValue(self._defaults["metatarsal"]["smoothing"])
self._meta_front_offset.setValue(self._defaults["metatarsal"]["front_offset"])
```

- [ ] **Step 5: `_reset_arch_params` を設定値を使うよう変更**

```python
def _reset_arch_params(self):
    self._arch_slider.setValue(self._defaults["arch"]["height_mm"])
    self._arch_smooth.setValue(self._defaults["arch"]["smoothing"])
```

- [ ] **Step 6: `_reset_meta_params` を設定値を使うよう変更**

```python
def _reset_meta_params(self):
    self._meta_height.setValue(self._defaults["metatarsal"]["height_mm"])
    self._meta_length.setValue(self._defaults["metatarsal"]["length_mm"])
    self._meta_width.setValue(self._defaults["metatarsal"]["width_mm"])
    self._meta_angle.setValue(self._defaults["metatarsal"]["angle_deg"])
    self._meta_smooth.setValue(self._defaults["metatarsal"]["smoothing"])
    self._meta_front_offset.setValue(self._defaults["metatarsal"]["front_offset"])
```

- [ ] **Step 7: `_open_settings` メソッドを追加**

`_reset_meta_params` の直後に追加：

```python
def _open_settings(self):
    dlg = SettingsDialog(self._defaults, parent=self)
    if dlg.exec() == QDialog.DialogCode.Accepted:
        values = dlg.get_values()
        _save_settings(values)
        self._defaults = values
```

- [ ] **Step 8: 構文チェック**

```bash
python -c "import ast; ast.parse(open('riki_footscan/main.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 9: 手動動作確認**

```
python riki_footscan/main.py
```

確認項目：
1. メニューバーに「設定」が表示される
2. 「設定 → 設定を開く...」でダイアログが開く
3. アーチ・メタターサルの各値を変更して「保存して閉じる」
4. アプリを再起動して設定値が保持されていること（スライダー・スピンボックスの初期値が変わっていること）
5. アーチタブの「設定値を初期化」が保存した値にリセットされること
6. メタターサルタブの「設定値を初期化」が保存した値にリセットされること
7. 「キャンセル」では設定が保存されないこと

- [ ] **Step 10: コミット**

```bash
git add riki_footscan/main.py
git commit -m "feat: アーチ・メタターサル初期値の設定ダイアログと保存機能を追加"
```

---

## 検証まとめ

| 検証項目 | 方法 |
|---|---|
| Ctrl+クリックでセル解除 | ドラッグ選択後にCtrl+クリックでハイライト消去 |
| Ctrl+クリックで未選択セル追加 | 未選択エリアにCtrl+クリックでハイライト追加 |
| 通常クリックは従来動作 | Ctrlなしクリック・ドラッグで追加選択 |
| 設定ダイアログ開閉 | メニューバー → 設定を開く... |
| 設定保存後の再起動 | アプリ再起動で初期値が反映 |
| リセットボタン | 保存値にリセットされること |
| キャンセルで保存されない | ダイアログキャンセル後に再起動して値が変わらないこと |
