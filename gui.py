#!/usr/bin/env python3
"""Brawlhalla AI — GUI controller. Thread-safe, dark theme, live preview."""

import sys
import os
import threading
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QCheckBox, QTextEdit, QGroupBox,
    QGridLayout, QComboBox, QFrame, QSizePolicy, QSpacerItem,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QColor, QFont, QPalette, QPixmap, QImage

from ai_runner import AIRunner
import game_state
from config import CHARACTER_PROFILES, CHARACTER


class AIWorkerSignals(QObject):
    """Thread-safe signals for UI updates from AI thread."""
    stats = pyqtSignal(float, float, int, bool, object)
    log_message = pyqtSignal(str)
    detection_preview = pyqtSignal(object)

class AIWorker:
    def __init__(self, signals: AIWorkerSignals):
        self.signals = signals
        self._runner: AIRunner | None = None
        self._thread: threading.Thread | None = None

    def start(self, mode: str, aggression: float, dry_run: bool):
        if self._thread and self._thread.is_alive():
            return
        self._runner = AIRunner(mode=mode, aggressive=aggression, dry_run=dry_run)
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._runner:
            self._runner.stop()

    def _run_loop(self):
        def _stats(fps, dt_ms, enemies, player_ok, ext_stats=None):
            self.signals.stats.emit(float(fps), float(dt_ms), int(enemies), bool(player_ok), ext_stats or {})

        def _log(msg):
            self.signals.log_message.emit(msg)

        self._runner.run_loop(stats_cb=_stats, log_cb=_log)


class BrawlhallaWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.signals = AIWorkerSignals()
        self.ai_worker = AIWorker(self.signals)
        self._preview_timer = QTimer()
        self._init_ui()
        self._apply_theme()
        self._connect_signals()

    # ── UI setup ─────────────────────────────────────────────
    def _init_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)
        self.setCentralWidget(central)

        # ── Top bar: title + mode ──
        top = QHBoxLayout()
        title = QLabel("BRAWLHALLA AI")
        title_font = QFont("Noto Sans", 16, QFont.Weight.Bold)
        title.setFont(title_font)

        self.mode_badge = QLabel("STOPPED")
        self.mode_badge.setStyleSheet(
            "background: #1e293b; color: #94a3b8; border-radius: 6px; "
            "padding: 4px 14px; font-weight: bold; font-size: 11px;"
        )

        top.addWidget(title)
        top.addStretch()
        top.addWidget(self.mode_badge)
        root.addLayout(top)

        # ── Main columns ──
        cols = QHBoxLayout()
        left = QVBoxLayout()
        right = QVBoxLayout()
        cols.addLayout(left, 1)
        cols.addLayout(right, 2)

        # ── LEFT panel ──
        mode_g = QGroupBox("MODE")
        ml = QVBoxLayout(mode_g)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Assist", "Poke (light only)", "Combat (atk only)", "Full Auto"])
        self.mode_combo.setCurrentIndex(2)
        self.mode_combo.setToolTip("Poke: light attacks only, in-range\nCombat: all attacks, you move\nFull Auto: AI controls everything")
        ml.addWidget(self.mode_combo)
        left.addWidget(mode_g)

        char_g = QGroupBox("CHARACTER")
        cl = QVBoxLayout(char_g)
        self.char_combo = QComboBox()
        chars = list(CHARACTER_PROFILES.keys())
        self.char_combo.addItems(sorted(chars))
        self.char_combo.setCurrentText(CHARACTER)
        self.char_combo.setToolTip("Select which legend the AI plays as")
        cl.addWidget(self.char_combo)

        self.char_info = QLabel("balanced | mixed | 0.50")
        self.char_info.setStyleSheet("color: #94a3b8; font-size: 10px;")
        cl.addWidget(self.char_info)
        self.char_combo.currentTextChanged.connect(self._on_char_changed)
        left.addWidget(char_g)

        agg_g = QGroupBox("AGGRESSION")
        al = QVBoxLayout(agg_g)
        self.agg_slider = QSlider(Qt.Orientation.Horizontal)
        self.agg_slider.setRange(10, 100)
        self.agg_slider.setValue(int(CHARACTER_PROFILES.get(CHARACTER, {}).get("aggression", 0.5) * 100))
        self.agg_label = QLabel()
        self.agg_slider.valueChanged.connect(self._on_agg)
        self._on_agg(self.agg_slider.value())
        al.addWidget(self.agg_slider)
        al.addWidget(self.agg_label)
        left.addWidget(agg_g)

        opt_g = QGroupBox("OPTIONS")
        ol = QVBoxLayout(opt_g)
        self.dry_check = QCheckBox("Dry Run (no keys)")
        self.dry_check.setToolTip("Print decisions without pressing keys")
        self.preview_check = QCheckBox("Show Preview")
        self.preview_check.setChecked(True)
        ol.addWidget(self.dry_check)
        ol.addWidget(self.preview_check)
        left.addWidget(opt_g)

        # Buttons
        btn = QHBoxLayout()
        self.btn_start = QPushButton("START")
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setEnabled(False)
        btn.addWidget(self.btn_start, 1)
        btn.addWidget(self.btn_stop, 1)
        left.addLayout(btn)

        left.addStretch()

        # ── RIGHT panel ──
        # Preview
        prev_g = QGroupBox("PREVIEW")
        pv = QVBoxLayout(prev_g)
        self.preview_label = QLabel("No frame")
        self.preview_label.setMinimumHeight(180)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("background: #020617; border-radius: 6px; color: #475569;")
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        pv.addWidget(self.preview_label)
        right.addWidget(prev_g)

        # Stats grid
        stats_g = QGroupBox("STATS")
        sl = QGridLayout(stats_g)
        sl.setSpacing(4)

        def _stat_row(label, row):
            k = QLabel(label)
            k.setStyleSheet("color: #94a3b8;")
            v = QLabel("--")
            v.setStyleSheet("font-weight: bold; font-size: 14px;")
            sl.addWidget(k, row, 0)
            sl.addWidget(v, row, 1)
            return v

        self.stat_fps = _stat_row("FPS", 0)
        self.stat_dt = _stat_row("Loop Time", 1)
        self.stat_enemies = _stat_row("Enemies", 2)
        self.stat_player = _stat_row("Player", 3)
        self.stat_prox = _stat_row("Range", 4)
        self.stat_combo = _stat_row("Combo", 5)

        right.addWidget(stats_g)

        # Log
        log_g = QGroupBox("LOG")
        ll = QVBoxLayout(log_g)
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMinimumHeight(80)
        ll.addWidget(self.log_widget)
        right.addWidget(log_g)

        root.addLayout(cols)

    # ── Theme ────────────────────────────────────────────────
    def _apply_theme(self):
        self.setWindowTitle("Brawlhalla AI")
        self.setMinimumSize(780, 560)
        self.resize(820, 620)

        p = QPalette()
        p.setColor(QPalette.ColorRole.Window, QColor("#060b14"))
        p.setColor(QPalette.ColorRole.WindowText, QColor("#e2e8f0"))
        p.setColor(QPalette.ColorRole.Button, QColor("#0f172a"))
        p.setColor(QPalette.ColorRole.ButtonText, QColor("#e2e8f0"))
        p.setColor(QPalette.ColorRole.Base, QColor("#020617"))
        p.setColor(QPalette.ColorRole.Text, QColor("#e2e8f0"))
        p.setColor(QPalette.ColorRole.Highlight, QColor("#3b82f6"))
        self.setPalette(p)

        bf = QFont("Noto Sans", 10)

        sp = QPalette()
        sp.setColor(QPalette.ColorRole.Button, QColor("#10b981"))
        sp.setColor(QPalette.ColorRole.ButtonText, QColor("#020617"))
        self.btn_start.setPalette(sp)
        self.btn_start.setFont(bf)
        self.btn_start.setMinimumHeight(36)

        rp = QPalette()
        rp.setColor(QPalette.ColorRole.Button, QColor("#ef4444"))
        rp.setColor(QPalette.ColorRole.ButtonText, QColor("#020617"))
        self.btn_stop.setPalette(rp)
        self.btn_stop.setFont(bf)
        self.btn_stop.setMinimumHeight(36)

        self.log_widget.setFont(QFont("JetBrains Mono", 9))

    # ── Connect signals ─────────────────────────────────────
    def _connect_signals(self):
        self.signals.stats.connect(self._on_stats)
        self.signals.log_message.connect(self._on_log)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)

        self._preview_timer.timeout.connect(self._capture_preview)
        self._preview_timer.start(2000)

    # ── Callbacks ───────────────────────────────────────────
    def _on_char_changed(self, name):
        p = CHARACTER_PROFILES.get(name, CHARACTER_PROFILES["balanced"])
        agg = p["aggression"]
        style = p["combo_style"]
        spot = p["sweet_spot"]
        self.char_info.setText(f"{style} | {spot:.2f} | agg {agg:.2f}")
        self.agg_slider.blockSignals(True)
        self.agg_slider.setValue(int(agg * 100))
        self.agg_slider.blockSignals(False)
        self.agg_label.setText(f"Aggression: {agg:.2f}")
        self._update_combo_label()
        os.environ["BH_CHARACTER"] = name

    def _on_agg(self, val):
        self.agg_label.setText(f"Aggression: {val / 100:.2f}")

    def _update_combo_label(self, mode=None):
        name = self.char_combo.currentText()
        p = CHARACTER_PROFILES.get(name, CHARACTER_PROFILES["balanced"])
        style = p["combo_style"]
        if not mode:
            # No running mode selected yet—just show base combo style
            self.stat_combo.setText(style)
            return
        label = style.capitalize()
        if mode in ("combat", "full"):
            label += f" / {p['aggression']:.0%}"
        else:
            label += " (limited)"
        self.stat_combo.setText(label)

    def _on_start(self):
        # Reset tracking state before starting a new run
        try:
            game_state.reset_tracking()
        except Exception:
            pass

        if self.ai_worker._thread and self.ai_worker._thread.is_alive():
            return

        idx = self.mode_combo.currentIndex()
        mode = {0: "assist", 1: "poke", 2: "combat", 3: "full"}.get(idx, "full")
        agg = self.agg_slider.value() / 100.0
        dry = self.dry_check.isChecked()
        char = self.char_combo.currentText()
        os.environ["BH_CHARACTER"] = char

        # Update combo label based on mode + character profile
        self._update_combo_label(mode)

        self.ai_worker.start(mode, agg, dry)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.mode_combo.setEnabled(False)
        self.char_combo.setEnabled(False)
        self.agg_slider.setEnabled(False)
        self.mode_badge.setText(mode.upper())
        self.mode_badge.setStyleSheet(
            "background: #166534; color: #4ade80; border-radius: 6px; "
            "padding: 4px 14px; font-weight: bold; font-size: 11px;"
        )
        self._preview_timer.start(1000)

    def _on_stop(self):
        self.ai_worker.stop()
        self.stat_combo.setText("--")
        QTimer.singleShot(600, self._reset_ui)

    def _reset_ui(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.mode_combo.setEnabled(True)
        self.char_combo.setEnabled(True)
        self.agg_slider.setEnabled(True)
        self.mode_badge.setText("STOPPED")
        self.mode_badge.setStyleSheet(
            "background: #1e293b; color: #94a3b8; border-radius: 6px; "
            "padding: 4px 14px; font-weight: bold; font-size: 11px;"
        )
        self._preview_timer.start(2000)

    def _on_stats(self, fps, dt_ms, enemies, player_ok, ext_stats=None):
        self.stat_fps.setText(f"{fps:.1f}")
        self.stat_dt.setText(f"{dt_ms:.0f} ms")
        self.stat_enemies.setText(str(enemies))
        # Range: derive from enemy count + fps, not loop time
        if enemies > 0 and fps < 25:
            self.stat_prox.setText("LOW (slow)")
        elif enemies > 0:
            self.stat_prox.setText("OK")
        else:
            self.stat_prox.setText("--")

    def _on_log(self, msg: str):
        c = self.log_widget.textCursor()
        c.movePosition(c.MoveOperation.End)
        # Trim to avoid memory bloat
        if self.log_widget.document().lineCount() > 200:
            self.log_widget.clear()
        c.insertText(msg + "\n")

    def _capture_preview(self):
        if not self.preview_check.isChecked():
            self.preview_label.setText("Preview off")
            return
        if not self.ai_worker._runner or not self.ai_worker._runner.running:
            self.preview_label.setText("Not running")
            return
        try:
            from screen_capture import capture_frame
            import cv2
            frame = capture_frame()
            if frame is None or frame.size == 0:
                self.preview_label.setText("No frame")
                return
            h, w = frame.shape[:2]
            scale = min(260 / w, 180 / h)
            nw, nh = int(w * scale), int(h * scale)
            resized = cv2.resize(frame, (nw, nh))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.preview_label.setPixmap(QPixmap.fromImage(img))
            self.preview_label.setText("")
        except Exception:
            self.preview_label.setText("No frame")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("BrawlhallaAI")
    w = BrawlhallaWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()