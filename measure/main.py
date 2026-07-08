"""計測UI（Phase 2A: StubScanner使用）"""
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QToolBar, QLineEdit, QPushButton, QLabel, QProgressBar,
    QFileDialog, QMessageBox, QStatusBar, QDialog, QDialogButtonBox,
    QGroupBox, QRadioButton, QButtonGroup, QFormLayout,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QAction

sys.path.insert(0, str(Path(__file__).parent.parent))
import grd_io
from heatmap_widget import HeatmapWidget
from measure.scanner import StubScanner, SerialScanner, ScanWorker
from measure.calibration import CalibrationManager
from measure.analysis import (
    left_right_ratio, front_back_ratio, contact_area,
    RIGHT_ROWS, LEFT_ROWS,
)
from measure.model import MeasurementSession, UndoStack

_DEFAULT_GRD = Path(__file__).parent.parent / "rei_1.grd"


@dataclass
class PatientForm:
    last_name: str = ""
    first_name: str = ""
    romaji: str = ""
    age: str = ""
    gender: str = "男性"
    shoe_size_cm: str = ""
    foot_size_left: str = ""
    foot_size_right: str = ""

    def display_name(self) -> str:
        full = f"{self.last_name} {self.first_name}".strip()
        return full or self.romaji or "（未入力）"

    def to_patient_info(self) -> grd_io.PatientInfo:
        # 年齢はGRDファイルのメタデータには対応する保存先がないためアプリ内一時情報とし、
        # grd_io.PatientInfo（ファイルへ保存される情報）には含めない。
        return grd_io.PatientInfo(
            name=self.romaji or self.display_name(),
            country="Japan",
            gender=self.gender,
            shoe_size=self.shoe_size_cm,
            foot_size_left=self.foot_size_left,
            foot_size_right=self.foot_size_right,
        )


class PatientInfoDialog(QDialog):
    def __init__(self, form: PatientForm | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("患者情報入力")
        self.setMinimumWidth(380)
        self._form = form or PatientForm()
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        grp = QGroupBox("お客様データ")
        form_layout = QFormLayout(grp)
        form_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)

        # 姓 / 名
        name_widget = QWidget()
        name_row = QHBoxLayout(name_widget)
        name_row.setContentsMargins(0, 0, 0, 0)
        self._last_name = QLineEdit(self._form.last_name)
        self._last_name.setPlaceholderText("姓")
        self._first_name = QLineEdit(self._form.first_name)
        self._first_name.setPlaceholderText("名")
        name_row.addWidget(self._last_name)
        name_row.addWidget(QLabel("　名"))
        name_row.addWidget(self._first_name)
        form_layout.addRow("姓:", name_widget)

        # ローマ字
        self._romaji = QLineEdit(self._form.romaji)
        self._romaji.setPlaceholderText("例: rei 1")
        form_layout.addRow("ローマ字:", self._romaji)

        # 年齢
        self._age = QLineEdit(self._form.age)
        self._age.setPlaceholderText("例: 35")
        self._age.setFixedWidth(80)
        form_layout.addRow("年齢:", self._age)

        # 性別
        gender_widget = QWidget()
        gender_row = QHBoxLayout(gender_widget)
        gender_row.setContentsMargins(0, 0, 0, 0)
        self._gender_male = QRadioButton("男性")
        self._gender_female = QRadioButton("女性")
        self._gender_group = QButtonGroup(self)
        self._gender_group.addButton(self._gender_male)
        self._gender_group.addButton(self._gender_female)
        if self._form.gender == "女性":
            self._gender_female.setChecked(True)
        else:
            self._gender_male.setChecked(True)
        gender_row.addWidget(self._gender_male)
        gender_row.addWidget(self._gender_female)
        gender_row.addStretch()
        form_layout.addRow("性別:", gender_widget)

        # 靴のサイズ
        shoe_widget = QWidget()
        shoe_row = QHBoxLayout(shoe_widget)
        shoe_row.setContentsMargins(0, 0, 0, 0)
        self._shoe_size = QLineEdit(self._form.shoe_size_cm)
        self._shoe_size.setPlaceholderText("25")
        self._shoe_size.setFixedWidth(70)
        shoe_row.addWidget(self._shoe_size)
        shoe_row.addWidget(QLabel("cm"))
        shoe_row.addStretch()
        form_layout.addRow("靴のサイズ（半角）:", shoe_widget)

        # 足のサイズ（左/右）
        foot_widget = QWidget()
        foot_row = QHBoxLayout(foot_widget)
        foot_row.setContentsMargins(0, 0, 0, 0)
        self._foot_left = QLineEdit(self._form.foot_size_left)
        self._foot_left.setPlaceholderText("38.5")
        self._foot_left.setFixedWidth(70)
        self._foot_right = QLineEdit(self._form.foot_size_right)
        self._foot_right.setPlaceholderText("39")
        self._foot_right.setFixedWidth(70)
        foot_row.addWidget(QLabel("左"))
        foot_row.addWidget(self._foot_left)
        foot_row.addWidget(QLabel("　右"))
        foot_row.addWidget(self._foot_right)
        foot_row.addStretch()
        form_layout.addRow("足のサイズ（半角）:", foot_widget)

        layout.addWidget(grp)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_form(self) -> PatientForm:
        return PatientForm(
            last_name=self._last_name.text().strip(),
            first_name=self._first_name.text().strip(),
            romaji=self._romaji.text().strip(),
            age=self._age.text().strip(),
            gender="女性" if self._gender_female.isChecked() else "男性",
            shoe_size_cm=self._shoe_size.text().strip(),
            foot_size_left=self._foot_left.text().strip(),
            foot_size_right=self._foot_right.text().strip(),
        )


class PatientInfoBar(QWidget):
    """患者情報と日時を表示するバー"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        self._name_label = QLabel("お客様氏名: —")
        self._name_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._name_label)

        layout.addStretch()

        self._info_label = QLabel("年齢: —　性別: —　靴: —　足: —")
        layout.addWidget(self._info_label)

        layout.addStretch()

        self._created_label = QLabel("データ作成日時: —")
        layout.addWidget(self._created_label)

        self._updated_label = QLabel("データ変更日時: —")
        layout.addWidget(self._updated_label)

    def update_patient(self, form: PatientForm) -> None:
        self._name_label.setText(f"お客様氏名: {form.display_name()} 様")
        parts = []
        if form.age:
            parts.append(f"年齢: {form.age}")
        if form.gender:
            parts.append(f"性別: {form.gender}")
        if form.shoe_size_cm:
            parts.append(f"靴: {form.shoe_size_cm}cm")
        foot = []
        if form.foot_size_left:
            foot.append(f"左{form.foot_size_left}")
        if form.foot_size_right:
            foot.append(f"右{form.foot_size_right}")
        if foot:
            parts.append(f"足: {'　'.join(foot)}")
        self._info_label.setText("　".join(parts) if parts else "")

    def update_created(self, dt: datetime) -> None:
        self._created_label.setText(f"データ作成日時: {dt.strftime('%m/%d/%y %H:%M:%S')}")

    def update_modified(self, dt: datetime) -> None:
        self._updated_label.setText(f"データ変更日時: {dt.strftime('%m/%d/%y %H:%M:%S')}")


class BalancePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        self._lr_bar, self._lr_label = self._add_row(layout, "左右比:")
        self._fr_right_bar, self._fr_right_label = self._add_row(layout, "前後比(右):")
        self._fr_left_bar, self._fr_left_label = self._add_row(layout, "前後比(左):")

        self._contact_label = QLabel("右接触: -- セル  左接触: -- セル")
        layout.addWidget(self._contact_label)

    @staticmethod
    def _add_row(parent_layout, label_text: str):
        row = QHBoxLayout()
        row.addWidget(QLabel(label_text))
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setTextVisible(False)
        row.addWidget(bar)
        lbl = QLabel("-- : --")
        row.addWidget(lbl)
        parent_layout.addLayout(row)
        return bar, lbl

    def update_stats(self, grid: np.ndarray) -> None:
        left_pct, right_pct = left_right_ratio(grid)
        self._lr_bar.setValue(int(right_pct))
        self._lr_label.setText(f"右 {right_pct:.0f}:{left_pct:.0f} 左")

        right_foot = grid[RIGHT_ROWS]
        left_foot = grid[LEFT_ROWS]

        fr_r, bk_r = front_back_ratio(right_foot)
        self._fr_right_bar.setValue(int(fr_r))
        self._fr_right_label.setText(f"前 {fr_r:.0f}:{bk_r:.0f} 後")

        fr_l, bk_l = front_back_ratio(left_foot)
        self._fr_left_bar.setValue(int(fr_l))
        self._fr_left_label.setText(f"前 {fr_l:.0f}:{bk_l:.0f} 後")

        r_area = contact_area(right_foot)
        l_area = contact_area(left_foot)
        self._contact_label.setText(f"右接触: {r_area} セル  左接触: {l_area} セル")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("riki 計測")

        self._grd_path = _DEFAULT_GRD
        self._calib = CalibrationManager()
        self._undo = UndoStack()
        self._current_session: MeasurementSession | None = None
        self._patient_form = PatientForm()
        self._created_at: datetime | None = None
        self._dirty = False
        self._worker: ScanWorker | None = None
        self._scanner: SerialScanner | StubScanner | None = None
        self._combined_grid = np.zeros((64, 16), dtype=np.float32)

        self._build_ui()
        QTimer.singleShot(300, self._auto_connect_sensor)

    def _build_ui(self) -> None:
        # ツールバー
        toolbar = QToolBar("計測ツール")
        self.addToolBar(toolbar)

        # センサー接続状態
        self._sensor_status = QLabel("センサー: 未接続")
        self._sensor_status.setStyleSheet("color: gray; margin: 0 6px;")
        toolbar.addWidget(self._sensor_status)

        toolbar.addSeparator()

        self._act_scan = QAction("計測 ▶", self)
        self._act_scan.triggered.connect(self._on_scan)
        toolbar.addAction(self._act_scan)

        self._act_calib = QAction("キャリブ", self)
        self._act_calib.triggered.connect(self._on_calibrate)
        toolbar.addAction(self._act_calib)

        self._act_calib_reset = QAction("キャリブリセット", self)
        self._act_calib_reset.triggered.connect(self._on_calibrate_reset)
        toolbar.addAction(self._act_calib_reset)

        toolbar.addSeparator()

        btn_patient = QPushButton("患者情報")
        btn_patient.clicked.connect(self._on_patient_info)
        toolbar.addWidget(btn_patient)

        toolbar.addSeparator()

        btn_open = QPushButton("ファイルを開く")
        btn_open.clicked.connect(self._on_open_file)
        toolbar.addWidget(btn_open)

        btn_save = QPushButton("保存")
        btn_save.clicked.connect(self._on_save_as)
        toolbar.addWidget(btn_save)

        btn_overwrite = QPushButton("上書き")
        btn_overwrite.clicked.connect(self._on_overwrite)
        toolbar.addWidget(btn_overwrite)

        # セントラルウィジェット
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(4)

        # 患者情報バー
        self._info_bar = PatientInfoBar()
        self._info_bar.setStyleSheet(
            "PatientInfoBar { background: #f0f0f0; border-bottom: 1px solid #ccc; }"
        )
        main_layout.addWidget(self._info_bar)

        # ヒートマップ行
        heatmap_row = QHBoxLayout()
        self._hw_right = HeatmapWidget("右足")
        self._hw_left = HeatmapWidget("左足")
        heatmap_row.addWidget(self._hw_left)
        heatmap_row.addWidget(self._hw_right)
        main_layout.addLayout(heatmap_row, stretch=1)

        # バランスパネル
        self._balance_panel = BalancePanel()
        main_layout.addWidget(self._balance_panel)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("スタブスキャナー準備完了")
        self.resize(960, 680)

    @pyqtSlot()
    def _auto_connect_sensor(self) -> None:
        """起動時にCOM5へ自動接続を試みる"""
        try:
            self._scanner = SerialScanner("COM5")
            self._sensor_status.setText("センサー: COM5 接続済")
            self._sensor_status.setStyleSheet("color: green; margin: 0 6px;")
            self.statusBar().showMessage("センサー接続完了 (COM5 / 19200bps)")
        except Exception:
            self._sensor_status.setText("センサー: 未接続")
            self._sensor_status.setStyleSheet("color: gray; margin: 0 6px;")
            self.statusBar().showMessage("センサー未接続 — ファイルを開くかセンサーを接続してください")

    @pyqtSlot()
    def _load_default(self) -> None:
        """起動時にデフォルトGRDを直接表示（キャリブなし）"""
        if _DEFAULT_GRD.exists():
            try:
                data = grd_io.load(str(_DEFAULT_GRD))
                now = datetime.now()
                self._current_session = MeasurementSession(
                    data.grid, data.grid.copy(), data.patient, _DEFAULT_GRD
                )
                self._hw_right.set_grid(data.grid[RIGHT_ROWS])
                self._hw_left.set_grid(data.grid[LEFT_ROWS])
                self._balance_panel.update_stats(data.grid)
                self._info_bar.update_created(now)
                self._info_bar.update_modified(now)
                self.statusBar().showMessage(f"読み込み: {_DEFAULT_GRD.name}")
            except Exception:
                pass

    @pyqtSlot()
    def _on_patient_info(self) -> None:
        dlg = PatientInfoDialog(self._patient_form, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._patient_form = dlg.get_form()
            self._info_bar.update_patient(self._patient_form)

    @pyqtSlot()
    def _on_open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "GRD / GR2 ファイルを開く", "",
            "Foot scan files (*.grd *.gr2);;All files (*)"
        )
        if not path:
            return
        try:
            self._grd_path = Path(path)
            self._scanner = StubScanner(path)
            # ファイルの患者情報で PatientForm を初期化
            data = grd_io.load(path)
            p = data.patient
            self._patient_form = PatientForm(
                romaji=p.name,
                # 年齢はGRDファイルに保存されないため引き継げない（空のまま）
                gender=p.gender or "男性",
                shoe_size_cm=p.shoe_size,
                foot_size_left=p.foot_size_left,
                foot_size_right=p.foot_size_right,
            )
            self._info_bar.update_patient(self._patient_form)
            # 保存済みファイルはキャリブなしでそのまま表示
            now = datetime.now()
            session = MeasurementSession(data.grid, data.grid.copy(), data.patient, Path(path))
            self._current_session = session
            self._hw_right.set_grid(data.grid[RIGHT_ROWS])
            self._hw_left.set_grid(data.grid[LEFT_ROWS])
            self._balance_panel.update_stats(data.grid)
            self._info_bar.update_created(now)
            self._info_bar.update_modified(now)
            self.statusBar().showMessage(f"ファイルを読み込みました: {self._grd_path.name}")
        except Exception as exc:
            QMessageBox.critical(self, "読み込みエラー", str(exc))

    @pyqtSlot()
    def _on_scan(self) -> None:
        if self._scanner is None:
            QMessageBox.information(self, "計測", "先に「ファイルを開く」でセンサーまたはファイルを選択してください。")
            return
        self._worker = ScanWorker(self._scanner)
        self._worker.scan_done.connect(self._on_scan_done)
        self._worker.scan_error.connect(self._on_scan_error)
        self._act_scan.setEnabled(False)
        self._worker.start()

    @pyqtSlot(object)
    def _on_scan_done(self, raw: np.ndarray) -> None:
        self._act_scan.setEnabled(True)
        calibrated = self._calib.apply(raw)

        if isinstance(self._scanner, SerialScanner):
            # cols 0-7 = 左足センサー, cols 8-15 = 右足センサー (SerialScanner仕様)
            # 8列を16列中央(cols 4-12)にゼロパディング → 縦長表示・ULTRAFOOT準拠
            left_raw  = calibrated[:, 0:8]    # 32×8
            right_raw = calibrated[:, 8:16]   # 32×8
            left_cal  = np.zeros((32, 16), dtype=np.float32)
            right_cal = np.zeros((32, 16), dtype=np.float32)
            left_cal[:, 4:12]  = left_raw
            right_cal[:, 4:12] = right_raw
            self._combined_grid[LEFT_ROWS]  = left_cal
            self._combined_grid[RIGHT_ROWS] = right_cal
        else:
            self._combined_grid[:] = calibrated

        patient = self._patient_form.to_patient_info()
        session = MeasurementSession(raw, self._combined_grid.copy(), patient, None)

        now = datetime.now()
        if self._current_session is not None:
            self._undo.push(self._current_session)
            self._info_bar.update_modified(now)
        else:
            self._created_at = now
            self._info_bar.update_created(now)
            self._info_bar.update_modified(now)

        self._current_session = session
        self._dirty = True
        self._hw_right.set_grid(self._combined_grid[RIGHT_ROWS])
        self._hw_left.set_grid(self._combined_grid[LEFT_ROWS])
        self._balance_panel.update_stats(self._combined_grid)
        name = self._patient_form.display_name()
        self.statusBar().showMessage(f"お客様氏名: {name} 様　　両足計測完了")

    @pyqtSlot(str)
    def _on_scan_error(self, msg: str) -> None:
        self._act_scan.setEnabled(True)
        QMessageBox.critical(self, "スキャンエラー", msg)

    @pyqtSlot()
    def _on_calibrate(self) -> None:
        if self._scanner is None:
            QMessageBox.information(self, "キャリブ", "スキャナーが接続されていません。")
            return
        self._calib.capture_baseline(self._scanner)
        self.statusBar().showMessage("キャリブレーション保存完了")

    @pyqtSlot()
    def _on_calibrate_reset(self) -> None:
        try:
            if not self._calib.has_baseline():
                self.statusBar().showMessage("ベースラインは存在しません")
                return
            reply = QMessageBox.question(
                self, "キャリブリセット", "ベースラインを削除してキャリブをリセットしますか？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._calib.reset_baseline()
                self.statusBar().showMessage("キャリブレーションをリセットしました")
        except Exception as exc:
            import traceback
            QMessageBox.critical(self, "キャリブリセットエラー", traceback.format_exc())

    @pyqtSlot()
    def _on_save_as(self) -> None:
        if self._current_session is None:
            QMessageBox.information(self, "保存", "計測データがありません。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "GRD 保存", "", "GRD ファイル (*.grd)")
        if not path:
            return
        self._save(path)

    @pyqtSlot()
    def _on_overwrite(self) -> None:
        if self._current_session is None:
            return
        if self._current_session.saved_path is None:
            self._on_save_as()
            return
        self._save(str(self._current_session.saved_path))

    def _save(self, path: str) -> None:
        s = self._current_session
        pf = self._patient_form
        meta: list[str] = [""] * 30
        meta[0] = f"_({pf.romaji or pf.display_name()})"
        meta[5] = "Japan"
        meta[11] = pf.shoe_size_cm
        meta[13] = pf.gender
        meta[20] = pf.foot_size_left
        meta[21] = pf.foot_size_right
        patient = pf.to_patient_info()
        grd_io.save(grd_io.GrdData(grid=s.calibrated_grid, patient=patient, raw_meta_lines=meta), path)
        self._current_session = MeasurementSession(
            s.raw_grid, s.calibrated_grid, patient, Path(path), s.timestamp
        )
        self._dirty = False
        self.statusBar().showMessage(f"保存: {path}")

    def closeEvent(self, event) -> None:
        if self._dirty:
            reply = QMessageBox.question(
                self, "確認", "未保存のデータがあります。終了しますか？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
