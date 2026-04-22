import sys
import os
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton,
    QFileDialog, QHBoxLayout, QLineEdit, QSpinBox, QGroupBox, QTextEdit, 
    QCheckBox, QFrame, QGridLayout, QProgressBar, QSizePolicy
)
from PyQt5.QtCore import QTimer, Qt, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt5.QtGui import QFont, QFontMetrics, QPalette, QColor, QPainter, QLinearGradient, QPen
import pyqtgraph as pg

import serial

APP_TITLE = "Real-Time Detection of Ganoderma Based on Radar"

FEATURE_ORDER_FALLBACK = [
    'ADC1_Mean','ADC1_Std','ADC1_Var','ADC1_Min','ADC1_Max','ADC1_Median','ADC1_Q25','ADC1_Q75',
    'ADC2_Mean','ADC2_Std','ADC2_Var','ADC2_Min','ADC2_Max','ADC2_Median','ADC2_Q25','ADC2_Q75',
    'Voltage1_Mean','Voltage1_Std','Voltage1_Var','Voltage1_Min','Voltage1_Max','Voltage1_Median','Voltage1_Q25','Voltage1_Q75',
    'Voltage2_Mean','Voltage2_Std','Voltage2_Var','Voltage2_Min','Voltage2_Max','Voltage2_Median','Voltage2_Q25','Voltage2_Q75',
]

# Status mapping
STATUS_MAPPING = {
    0: "Healthy",
    1: "Mild Infection",
    2: "Severe Infection",
    "healthy": "Healthy",
    "mild": "Mild Infection",
    "severe": "Severe Infection",
    "sehat": "Healthy",
    "ringan": "Mild Infection",
    "berat": "Severe Infection",
}

STATUS_COLORS = {
    "Healthy": "#4CAF50",    # Green
    "Mild Infection": "#E48900",   # Orange
    "Severe Infection": "#F44336",    # Red
    "—": "#9E9E9E"         # Gray
}

def quantile(arr, q):
    if len(arr) == 0:
        return np.nan
    return float(np.quantile(arr, q))

def compute_features(adc1, adc2, v1, v2):
    def stats(x):
        x = np.asarray(x, dtype=float)
        return {
            "Mean": float(np.nanmean(x)),
            "Std": float(np.nanstd(x, ddof=1)) if len(x) > 1 else 0.0,
            "Var": float(np.nanvar(x, ddof=1)) if len(x) > 1 else 0.0,
            "Min": float(np.nanmin(x)),
            "Max": float(np.nanmax(x)),
            "Median": float(np.nanmedian(x)),
            "Q25": quantile(x, 0.25),
            "Q75": quantile(x, 0.75),
        }
    s1 = stats(adc1); s2 = stats(adc2); s3 = stats(v1); s4 = stats(v2)
    feats = {
        'ADC1_Mean': s1["Mean"], 'ADC1_Std': s1["Std"], 'ADC1_Var': s1["Var"],
        'ADC1_Min': s1["Min"], 'ADC1_Max': s1["Max"], 'ADC1_Median': s1["Median"],
        'ADC1_Q25': s1["Q25"], 'ADC1_Q75': s1["Q75"],
        'ADC2_Mean': s2["Mean"], 'ADC2_Std': s2["Std"], 'ADC2_Var': s2["Var"],
        'ADC2_Min': s2["Min"], 'ADC2_Max': s2["Max"], 'ADC2_Median': s2["Median"],
        'ADC2_Q25': s2["Q25"], 'ADC2_Q75': s2["Q75"],
        'Voltage1_Mean': s3["Mean"], 'Voltage1_Std': s3["Std"], 'Voltage1_Var': s3["Var"],
        'Voltage1_Min': s3["Min"], 'Voltage1_Max': s3["Max"], 'Voltage1_Median': s3["Median"],
        'Voltage1_Q25': s3["Q25"], 'Voltage1_Q75': s3["Q75"],
        'Voltage2_Mean': s4["Mean"], 'Voltage2_Std': s4["Std"], 'Voltage2_Var': s4["Var"],
        'Voltage2_Min': s4["Min"], 'Voltage2_Max': s4["Max"], 'Voltage2_Median': s4["Median"],
        'Voltage2_Q25': s4["Q25"], 'Voltage2_Q75': s4["Q75"],
    }
    return feats

def align_features(df, feature_names):
    X = df.copy()
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_names]
    X = X.fillna(X.median(numeric_only=True))
    return X

class AnimatedLabel(QLabel):
    def __init__(self, text=""):
        super().__init__(text)
        self._color = QColor(0, 0, 0)
        
    def get_color(self):
        return self._color
    
    def set_color(self, color):
        self._color = color
        self.setStyleSheet(f"color: {color.name()};")
    
    color = pyqtProperty(QColor, get_color, set_color)

class StatusIndicator(QFrame):
    def __init__(self):
        super().__init__()
        self.setFixedSize(120, 120)
        self.status = "—"
        self.confidence = 0.0
        
    def set_status(self, status, confidence=0.0):
        self.status = status
        self.confidence = confidence
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw circle background
        color = QColor(STATUS_COLORS.get(self.status, "#9E9E9E"))
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(10, 10, 100, 100)
        
        # Draw confidence arc if available
        if self.confidence > 0:
            painter.setPen(QPen(QColor(255, 255, 255), 4))
            painter.setBrush(Qt.NoBrush)
            start_angle = 90 * 16  # Start from top
            span_angle = int(360 * self.confidence * 16)
            painter.drawArc(15, 15, 90, 90, start_angle, -span_angle)

class ModernGanodermaGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        # Get screen size and set window to fill most of it
        screen = QApplication.primaryScreen().geometry()
        self.setMinimumSize(int(screen.width() * 0.8), int(screen.height() * 0.8))
        self.resize(int(screen.width() * 0.95), int(screen.height() * 0.9))
        self.setStyleSheet(self.get_stylesheet())
        
        # Initialize variables
        self.serial = None
        self.timer_read = QTimer(self)
        self.timer_read.setInterval(1)
        self.timer_read.timeout.connect(self.read_serial)
        
        self.timer_plot = QTimer(self)
        self.timer_plot.setInterval(50)
        self.timer_plot.timeout.connect(self.update_plot)
        
        # RX buffer and protocol
        self.rxbuf = bytearray()
        self.PACK = 5
        self.HEADER = ord('e')
        
        # Acquisition parameters
        self.fs = 512
        self.window_seconds = 5
        self.window_samples = int(self.fs * self.window_seconds)
        
        # Data buffers
        self.time_s = []
        self.volt1 = []
        self.volt2 = []
        self.plot_win_seconds = 2.0
        self.sample_counter = 0  # persistent counter for time axis
        
        # Current window accumulators
        self.win_adc1 = []
        self.win_adc2 = []
        self.win_v1 = []
        self.win_v2 = []
        self.win_count = 0
        
        # Model variables
        self.bundle = None
        self.pipeline = None
        self.classes = None
        self.feature_names = None
        
        self.init_ui()
        
    def get_stylesheet(self):
        return """
        QMainWindow {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #033f63, stop:0.3 #28666e, stop:0.6 #7c9885, stop:0.8 #b5b682, stop:1 #fedc97);
        }
        
        QGroupBox {
    font-weight: bold;
    border: 2px solid #033f63;
    border-radius: 10px;
    margin-top: 12px;
    padding-top: 18px; /* allow bigger title without covering content */
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 rgba(255,255,255,0.95), stop:1 rgba(254,220,151,0.7));
}
        
        QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    top: 0px;
    padding: 3px 12px;
    background: rgba(255,255,255,0.96);
    border: 2px solid #033f63;
    border-radius: 10px;
    color: #033f63;
    font-size: 16px;   /* enlarged */
    font-weight: bold;
}
        
        /* Special style for detection result panel */
        QGroupBox#detection_panel {
            font-size: 22px;
            border: 4px solid #28666e;
            border-radius: 15px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 rgba(255,255,255,0.98), stop:1 rgba(124,152,133,0.3));
            margin-top: 20px;
            padding-top: 25px;
        }
        
        QGroupBox#detection_panel::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 26px;
            top: 8px;               /* slightly lower for big header */
            padding: 6px 16px;
            background: rgba(255,255,255,0.98);
            border: 3px solid #28666e;
            border-radius: 12px;
            font-size: 22px;
            font-weight: 800;
            color: #28666e;
        }
        
        QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #7c9885, stop:1 #28666e);
    border: none;
    border-radius: 8px;
    color: white;
    font-weight: bold;
    padding: 8px 14px;
    min-width: 90px;
    font-size: 12px;
    min-height: 16px;
}
        
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #28666e, stop:1 #033f63);
        }
        
        QPushButton:pressed {
            background: #033f63;
        }
        
        QPushButton:disabled {
            background: #bdbdbd;
            color: #757575;
        }
        
        QLineEdit, QSpinBox {
    border: 2px solid #7c9885;
    border-radius: 6px;
    padding: 6px;
    background: rgba(255, 255, 255, 0.9);
    selection-background-color: #b5b682;
    font-size: 12px;
    min-height: 18px;
}
        
        QLineEdit:focus, QSpinBox:focus {
            border-color: #28666e;
        }
        
        QTextEdit {
            border: 3px solid #7c9885;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.95);
            font-family: 'Consolas', monospace;
            font-size: 13px;
            padding: 8px;
        }
        
        QProgressBar {
    border: 2px solid #7c9885;
    border-radius: 6px;
    text-align: center;
    background: rgba(255, 255, 255, 0.9);
    font-size: 11px;
    min-height: 20px;
}
        
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #28666e, stop:1 #7c9885);
            border-radius: 4px;
        }
        
        QLabel {
    color: #033f63;
    font-size: 14px;
    font-weight: 500;
}
        
        QCheckBox {
            font-weight: bold;
            color: #033f63;
            font-size: 15px;
        }
        
        QCheckBox::indicator:checked {
            background: #7c9885;
            border: 2px solid #28666e;
        }
        

/* === OVERRIDE+++ (FINAL): Bigger titles for 5 sections === */
QGroupBox#conn_panel,
QGroupBox#params_panel,
QGroupBox#status_panel,
QGroupBox#log_panel,
QGroupBox#signal_panel {
    padding-top: 32px;  /* ensure enough space for the larger title */
    border-width: 2px;
    border-radius: 12px;
}

"""
        
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout - Full width approach
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Header - Much larger and more prominent
        header_label = QLabel(APP_TITLE)
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setFont(QFont("Arial", 24, QFont.Bold))
        header_label.setStyleSheet("""
            color: #033f63; 
            margin: 15px; 
            padding: 20px; 
            font-size: 24px;
            font-weight: bold;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 rgba(255,255,255,0.95), stop:1 rgba(254,220,151,0.8));
            border-radius: 12px;
            border: 3px solid #28666e;
        """)
        header_label.setWordWrap(True)
        header_label.setMinimumHeight(100)
        main_layout.addWidget(header_label)
        
        # Detection Result - Full width panel
        self.create_detection_panel(main_layout)
        
        # Middle section - Controls and small plot
        middle_layout = QHBoxLayout()
        middle_layout.setSpacing(15)
        
        # Left side controls
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(12)
        
        # Connection Controls
        self.create_connection_panel(controls_layout)
        
        # System Parameters
        self.create_parameters_panel(controls_layout)
        
        # Progress and Status
        self.create_status_panel(controls_layout)
        
        controls_layout.addStretch()
        
        # Right side - smaller plot
        plot_layout = QVBoxLayout()
        self.create_plot_panel(plot_layout)
        
        # Add to middle layout - give more space to plot
        middle_layout.addLayout(controls_layout, 1)  # slimmer controls panel
        middle_layout.addLayout(plot_layout, 4)      # wider real-time signal area
        middle_layout.setStretch(0, 1)
        middle_layout.setStretch(1, 4)
        
        main_layout.addLayout(middle_layout)
        
        # Bottom - Log panel with larger text
        self.create_log_panel(main_layout)
        
    def create_detection_panel(self, parent_layout):
        detection_group = QGroupBox("🔍 Detection Result")

        detection_group.setFont(QFont('Segoe UI', 11, QFont.Bold))
        detection_group.setObjectName("detection_panel")
        layout = QHBoxLayout(detection_group)  # Changed to horizontal for full width
        layout.setSpacing(30)
        
        # Status indicator on the left
        self.status_indicator = StatusIndicator()
        self.status_indicator.setFixedSize(150, 150)
        layout.addWidget(self.status_indicator, alignment=Qt.AlignLeft)
        
        # Status text in center
        status_text_layout = QVBoxLayout()
        
        self.status_label = QLabel("—")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Arial", 46, QFont.Black))
        self.status_label.setStyleSheet("color: #9e9e9e; margin: 10px;")
        status_text_layout.addWidget(self.status_label)
        
        self.confidence_label = QLabel("Confidence: —")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setFont(QFont("Arial", 24))  # bigger
        self.confidence_label.setStyleSheet(
            "color: #ffffff; margin: 6px; font-weight: 700;"
            "background: #033f63; padding: 6px 12px; border-radius: 10px;"
        )
        status_text_layout.addWidget(self.confidence_label)
        
        layout.addLayout(status_text_layout, 2)
        
        # Info on the right
        info_layout = QGridLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setHorizontalSpacing(6)
        info_layout.setVerticalSpacing(4)
        info_layout.setColumnStretch(0, 0)
        info_layout.setColumnStretch(1, 0)
        info_layout.setVerticalSpacing(8)  # Increased spacing
        info_layout.setHorizontalSpacing(2)
        
        model_label = QLabel("Model:")
        model_label.setFont(QFont("Arial", 18, QFont.Bold))  # Increased from 16
        model_label.setStyleSheet("color: #033f63; font-weight: bold;")  # More contrast
        info_layout.addWidget(model_label, 0, 0)
        
        self.model_status = QLabel("Not loaded")
        self.model_status.setStyleSheet("font-weight: 600; color:#fff; font-size: 14px;"
    "background:#43a047; padding: 4px 10px; border-radius: 6px;")  # White text on red background
        info_layout.addWidget(self.model_status, 0, 1)
        
        status_label = QLabel("Status:")
        status_label.setFont(QFont("Arial", 18, QFont.Bold))  # Increased from 16
        status_label.setStyleSheet("color: #033f63; font-weight: bold;")  # More contrast
        info_layout.addWidget(status_label, 1, 0)
        
        self.connection_status = QLabel("Disconnected")
        self.connection_status.setStyleSheet("font-weight: 600; color:#fff; font-size: 16px;"
    "background:#e53935; padding: 4px 10px; border-radius: 6px;")  # White text on red background
        info_layout.addWidget(self.connection_status, 1, 1)
        
        info_widget = QWidget()
        info_widget.setLayout(info_layout)
        info_widget.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        layout.addWidget(info_widget, alignment=Qt.AlignLeft)
        
        parent_layout.addWidget(detection_group)
        
    def create_connection_panel(self, parent_layout):
        conn_group = QGroupBox("📡 Radar Connection")

        conn_group.setFont(QFont('Segoe UI', 11, QFont.Bold))
        conn_group.setStyleSheet("""
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    top: 0 px;               /* push down to avoid clipping */
    padding: 4px 12px;        /* compact pill */
    background: rgba(255,255,255,0.96);
    border: 2px solid #033f63;
    border-radius: 10px;
    color: #033f63;
    font-size: 11px;          /* requested size */
    font-weight: 800;
}
QGroupBox { padding-top: 28px; }
""")
        conn_group.setObjectName('conn_panel')
        layout = QGridLayout(conn_group)
        layout.setContentsMargins(12, 28, 12, 12)
        layout.setAlignment(Qt.AlignTop)
        layout.setVerticalSpacing(8)
        layout.setHorizontalSpacing(10)
        layout.setRowStretch(2, 1)  # push content to top

        layout.setContentsMargins(8, 10, 8, 8)
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 0)
        layout.setColumnStretch(2, 0)
        layout.setColumnStretch(3, 0)
        layout.setColumnStretch(4, 1)

        
        # Port and baud with larger inputs
        port_label = QLabel("Port:")
        port_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(port_label, 0, 0)
        
        self.port_edit = QLineEdit("COM4")
        self.port_edit.setMinimumWidth(120)
        layout.addWidget(self.port_edit, 0, 1)
        
        baud_label = QLabel("Baud:")
        baud_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(baud_label, 0, 2)
        
        self.baud_spin = QSpinBox()
        self.baud_spin.setRange(9600, 921600)
        self.baud_spin.setValue(115200)
        self.baud_spin.setMinimumWidth(120)
        layout.addWidget(self.baud_spin, 0, 3)
        
        # Control buttons - larger
        button_layout = QHBoxLayout()
        self.connect_btn = QPushButton("🔌 Connect")
        self.connect_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.connect_btn.setMinimumWidth(120)
        self.connect_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.connect_btn.clicked.connect(self.toggle_connection)
        
        self.start_btn = QPushButton("▶️ Start")
        self.start_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.start_btn.setMinimumWidth(110)
        self.start_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.start_btn.clicked.connect(self.toggle_detection)
        self.start_btn.setEnabled(False)
        
        button_layout.addStretch(1)
        button_layout.addSpacing(10)
        button_layout.addWidget(self.connect_btn)
        button_layout.addStretch(2)
        button_layout.addWidget(self.start_btn)
        
        button_layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(button_layout, 1, 0, 1, 4)
        layout.setAlignment(button_layout, Qt.AlignTop | Qt.AlignRight)
        
        parent_layout.addWidget(conn_group)
        
    def create_parameters_panel(self, parent_layout):
        params_group = QGroupBox("🛠️ System Parameter")

        params_group.setFont(QFont('Segoe UI', 11, QFont.Bold))
        params_group.setStyleSheet("""
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    top: 0px;               /* push down to avoid clipping */
    padding: 4px 12px;        /* compact pill */
    background: rgba(255,255,255,0.96);
    border: 2px solid #033f63;
    border-radius: 10px;
    color: #033f63;
    font-size: 11px;          /* requested size */
    font-weight: 800;
}
QGroupBox { padding-top: 28px; }
""")
        params_group.setObjectName('params_panel')
        layout = QGridLayout(params_group)
        layout.setContentsMargins(12, 28, 12, 12)
        layout.setAlignment(Qt.AlignTop)
        layout.setVerticalSpacing(8)
        layout.setHorizontalSpacing(10)
        layout.setRowStretch(2, 1)  # push content to top

        layout.setContentsMargins(8, 10, 8, 8)
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 0)
        layout.setColumnStretch(2, 0)
        layout.setColumnStretch(3, 0)
        layout.setColumnStretch(4, 1)

        
        layout.addWidget(QLabel("Sampling (Hz):"), 0, 0)
        self.fs_spin = QSpinBox()
        self.fs_spin.setRange(100, 4096)
        self.fs_spin.setValue(self.fs)
        self.fs_spin.setMinimumWidth(80)
        self.fs_spin.valueChanged.connect(self.update_fs)
        layout.addWidget(self.fs_spin, 0, 1)
        
        layout.addWidget(QLabel("Window (s):"), 0, 2)
        self.window_spin = QSpinBox()
        self.window_spin.setRange(1, 30)
        self.window_spin.setValue(self.window_seconds)
        self.window_spin.setMinimumWidth(80)
        self.window_spin.valueChanged.connect(self.update_window)
        layout.addWidget(self.window_spin, 0, 3)
        
        # Model loading
        self.load_model_btn = QPushButton("📁 Load Model")
        self.load_model_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.load_model_btn.setMinimumWidth(140)
        self.load_model_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.load_model_btn.clicked.connect(self.load_model_dialog)
        layout.addWidget(self.load_model_btn, 1, 0, 1, 4)
        
        parent_layout.addWidget(params_group)
        
    

    def _set_progress_label(self, count, total):
        """Set small progress label right of the bar; wrap to 2 lines if long."""
        try:
            count_i = int(count)
            total_i = int(total)
        except Exception:
            count_i, total_i = count, total
        s = f"{count_i}/{total_i}"
        if len(str(total_i)) >= 4 or len(s) > 5:
            s = f"{count_i}/{total_i}"
        if hasattr(self, "progress_label") and self.progress_label is not None:
            self.progress_label.setWordWrap(False)
            self.progress_label.setText(s)
        return s
    def create_status_panel(self, parent_layout):
        status_group = QGroupBox("📊 Processing Status")

        status_group.setFont(QFont('Segoe UI', 11, QFont.Bold))
        status_group.setObjectName('status_panel')
        status_group.setStyleSheet("""
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    top: 0px;               /* push down to avoid clipping */
    padding: 4px 12px;        /* compact pill */
    background: rgba(255,255,255,0.96);
    border: 2px solid #033f63;
    border-radius: 10px;
    color: #033f63;
    font-size: 11px;          /* requested size */
    font-weight: 800;
}
QGroupBox { padding-top: 28px; }
""")
        layout = QVBoxLayout(status_group)
        layout.setContentsMargins(12, 28, 12, 12)
        layout.setSpacing(12)
        
        # Progress bar
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Progress:"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(self.window_samples)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar, 1)
        progress_layout.setSpacing(6)
        
        self.progress_label = QLabel("0/0")
        self.progress_label.setWordWrap(False)
        self.progress_label.setFont(QFont("Arial", 9))
        digits = max(2, len(str(self.window_samples)))
        sample_text = ('9' * digits) + '/' + ('9' * digits)
        fm = QFontMetrics(self.progress_label.font())
        self.progress_label.setFixedWidth(fm.horizontalAdvance(sample_text) + 8)
        self.progress_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        fm = QFontMetrics(self.progress_label.font())
        self.progress_label.setFixedHeight(fm.height() + 6)
        self.progress_label.setStyleSheet("font-size: 11px; padding: 1px 4px;")
        progress_layout.setAlignment(self.progress_label, Qt.AlignVCenter)
        progress_layout.addWidget(self.progress_label)
        
        layout.addLayout(progress_layout)
        
        # Statistics
        stats_layout = QGridLayout()
        stats_layout.setVerticalSpacing(8)
        stats_layout.setHorizontalSpacing(10)
        
        stats_layout.addWidget(QLabel("Total Detections:"), 0, 0)
        self.total_detections = QLabel("0")
        self.total_detections.setStyleSheet("font-weight: bold; font-size: 14px;")
        stats_layout.addWidget(self.total_detections, 0, 1)
        
        stats_layout.addWidget(QLabel("Rate (detections/min):"), 0, 2)
        self.detection_rate = QLabel("0.0")
        self.detection_rate.setStyleSheet("font-weight: bold; font-size: 14px;")
        stats_layout.addWidget(self.detection_rate, 0, 3)
        
        layout.addLayout(stats_layout)
        
        parent_layout.addWidget(status_group)
        
    def create_log_panel(self, parent_layout):
        log_group = QGroupBox("📋 System Log")

        log_group.setFont(QFont('Segoe UI', 11, QFont.Bold))
        log_group.setObjectName('log_panel')
        log_group.setStyleSheet("""
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    top: 0px;               /* push down to avoid clipping */
    padding: 4px 12px;        /* compact pill */
    background: rgba(255,255,255,0.96);
    border: 2px solid #033f63;
    border-radius: 10px;
    color: #033f63;
    font-size: 11px;          /* requested size */
    font-weight: 800;
}
QGroupBox { padding-top: 28px; }
""")
        layout = QVBoxLayout(log_group)
        
        layout.setContentsMargins(12, 28, 12, 12)
        self.log_text = QTextEdit()
        self.log_text.setMinimumHeight(180)
        self.log_text.setMaximumHeight(220)
        self.log_text.setReadOnly(True)
        # Larger font for log
        self.log_text.setStyleSheet("""
            QTextEdit {
                border: 3px solid #7c9885;
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.95);
                font-family: 'Consolas', monospace;
                font-size: 14px;
                padding: 8px;
            }
        """)
        layout.addWidget(self.log_text)
        
        parent_layout.addWidget(log_group)
        
    def create_plot_panel(self, parent_layout):
        plot_group = QGroupBox("📈 Real-Time Signal")

        plot_group.setFont(QFont('Segoe UI', 11, QFont.Bold))
        plot_group.setStyleSheet("""
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    top: 0px;               /* push down to avoid clipping */
    padding: 4px 12px;        /* compact pill */
    background: rgba(255,255,255,0.96);
    border: 2px solid #033f63;
    border-radius: 10px;
    color: #033f63;
    font-size: 11px;          /* requested size */
    font-weight: 800;
}
QGroupBox { padding-top: 28px; }
""")
        plot_group.setObjectName('signal_panel')
        plot_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout = QVBoxLayout(plot_group)
        layout.setContentsMargins(12, 28, 12, 12)
        layout.setSpacing(10)
        
        # Plot controls - larger
        plot_controls = QHBoxLayout()
        self.plot_enabled = QCheckBox("Show Plot")
        self.plot_enabled.setChecked(True)
        self.plot_enabled.setFont(QFont("Arial", 13, QFont.Bold))
        plot_controls.addWidget(self.plot_enabled)
        plot_controls.addStretch()
        layout.addLayout(plot_controls)
        
        # Plot widget - smaller size
        self.plot_widget = pg.PlotWidget(background='w')
        # self.plot_widget.setLabel('left', 'Voltage (V)', color="#000000", size='13pt')
        # self.plot_widget.setLabel('bottom', 'Time (s)', color="#000000", size='13pt')
        self.plot_widget.setLabel('left', '<b>Voltage (V)</b>', color="#28666e", size='15pt')
        self.plot_widget.setLabel('bottom', '<b>Time (s)</b>', color="#28666e", size='15pt')    
        self.plot_widget.addLegend()
        self.plot_widget.setMinimumHeight(250)  # Reduced height
        self.plot_widget.setMaximumHeight(350)  # Set maximum height
        
        pen1 = pg.mkPen('#28666e', width=2)
        self.curve1 = self.plot_widget.plot(pen=pen1, name="Channel 1")
        pen2 = pg.mkPen('#ff7f0e', width=2)
        self.curve2 = self.plot_widget.plot(pen=pen2, name="Channel 2")
        
        layout.addWidget(self.plot_widget)
        parent_layout.addWidget(plot_group)
        
    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
    def update_fs(self, value):
        self.fs = value
        self.window_samples = int(self.fs * self.window_seconds)
        self.progress_bar.setMaximum(self.window_samples)
        self.log_message(f"Sampling rate: {self.fs} Hz")
        
    def update_window(self, value):
        self.window_seconds = value
        self.window_samples = int(self.fs * self.window_seconds)
        self.progress_bar.setMaximum(self.window_samples)
        self.log_message(f"Window size: {self.window_seconds} s")
        
    def load_model_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "Pickle Files (*.pkl);;All Files (*)"
        )
        if file_path:
            self.load_model(file_path)
            
    def load_model(self, file_path):
        try:
            bundle = joblib.load(file_path)
            if isinstance(bundle, dict) and "pipeline" in bundle:
                self.pipeline = bundle["pipeline"]
                self.classes = bundle.get("classes", None)
                self.feature_names = bundle.get("feature_names", None)
            else:
                self.pipeline = bundle
                self.classes = None
                self.feature_names = None
                
            if self.classes is None and hasattr(self.pipeline, "classes_"):
                self.classes = list(getattr(self.pipeline, "classes_"))
                
            if self.feature_names is None:
                self.feature_names = FEATURE_ORDER_FALLBACK
                
            self.bundle = bundle
            
            model_name = os.path.basename(file_path)
            self.model_status.setText("Ready")
            self.model_status.setStyleSheet("font-weight: bold; color: #ffffff; font-size: 20px; background: #4caf50; padding: 6px 12px; border-radius: 8px;")  # White text on green
            self.log_message(f"Model loaded: {model_name}")
            
            if self.serial is not None:
                self.start_btn.setEnabled(True)
                
        except Exception as e:
            self.log_message(f"Error loading model: {str(e)}")
            
    def toggle_connection(self):
        if self.serial is None:
            try:
                port = self.port_edit.text().strip()
                baud = self.baud_spin.value()
                
                self.serial = serial.Serial(
                    port, baudrate=baud, timeout=0,
                    write_timeout=0, inter_byte_timeout=0,
                    dsrdtr=False, rtscts=False
                )
                
                self.serial.reset_input_buffer()
                self.rxbuf.clear()
                
                self.connect_btn.setText("🔌 Disconnect")
                self.connection_status.setText("Connected")
                self.connection_status.setStyleSheet("font-weight: bold; color: #ffffff; font-size: 20px; background: #4caf50; padding: 6px 12px; border-radius: 8px;")  # White text on green
                
                if self.pipeline is not None:
                    self.start_btn.setEnabled(True)
                    
                if self.plot_enabled.isChecked():
                    self.timer_plot.start()
                    
                self.log_message(f"Connected to {port} @ {baud} baud")
                
            except Exception as e:
                self.log_message(f"Connection failed: {str(e)}")
        else:
            self.disconnect_serial()
            
    def disconnect_serial(self):
        if self.serial:
            try:
                self.timer_read.stop()
                self.timer_plot.stop()
                self.serial.close()
            except:
                pass
                
            self.serial = None
            self.connect_btn.setText("🔌 Connect")
            self.start_btn.setEnabled(False)
            self.start_btn.setText("▶️ Start")
            
            self.connection_status.setText("Disconnected")
            self.connection_status.setStyleSheet("font-weight: bold; color: #ffffff; font-size: 18px; background: #f44336; padding: 5px 10px; border-radius: 5px;")  # White text on red
            
            self.log_message("Disconnected")
            
    def toggle_detection(self):
        if not self.timer_read.isActive():
            if self.serial is None or self.pipeline is None:
                self.log_message("Pastikan radar terhubung dan model sudah dimuat")
                return
                
            try:
                self.serial.write(b'a')  # Start command
                self.reset_buffers()
                self.timer_read.start()
                
                self.start_btn.setText("⏹️ Stop")
                self.log_message("Detection started")
                
            except Exception as e:
                self.log_message(f"Start failed: {str(e)}")
        else:
            try:
                self.serial.write(b'b')  # Stop command
                self.timer_read.stop()
                
                self.start_btn.setText("▶️ Mulai")
                self.log_message("Detection stopped")
                
            except Exception as e:
                self.log_message(f"Stop failed: {str(e)}")
                
    def reset_buffers(self):
        self.time_s.clear()
        self.volt1.clear()
        self.volt2.clear()
        self.win_adc1.clear()
        self.win_adc2.clear()
        self.win_v1.clear()
        self.win_v2.clear()
        self.win_count = 0
        self.sample_counter = 0
        self.rxbuf.clear()
        self.progress_bar.setValue(0)
        self._set_progress_label(0, 0)
        
    def parse_packets_from_buffer(self):
        packets = []
        buf = self.rxbuf
        i = 0
        n = len(buf)
        
        while True:
            if n - i < self.PACK:
                break
                
            if buf[i] != self.HEADER:
                try:
                    j = buf.index(self.HEADER, i + 1, n)
                    i = j
                except ValueError:
                    keep_from = max(0, n - (self.PACK - 1))
                    del buf[:keep_from]
                    return packets
                    
            if n - i < self.PACK:
                if i > 0:
                    del buf[:i]
                return packets
                
            a1, a2, b1, b2 = buf[i + 1:i + 5]
            adc1 = a1 * 100 + a2
            adc2 = b1 * 100 + b2
            v1 = (adc1 / 4095.0) * 3.3
            v2 = (adc2 / 4095.0) * 3.3
            
            packets.append((adc1, adc2, v1, v2))
            i += self.PACK
            
        if i > 0:
            del buf[:i]
            
        return packets
        
    def read_serial(self):
        try:
            if hasattr(self.serial, 'in_waiting'):
                n_bytes = self.serial.in_waiting
                if n_bytes > 0:
                    data = self.serial.read(n_bytes)
                    if data:
                        self.rxbuf.extend(data)
                        
            packets = self.parse_packets_from_buffer()
            if not packets:
                return
                
            for adc1, adc2, v1, v2 in packets:
                # Add to plot buffers
                t = (self.time_s[-1] + (1.0/self.fs)) if self.time_s else 0.0
                self.time_s.append(t)
                self.volt1.append(v1)
                self.volt2.append(v2)
                
                # Add to window buffers
                self.win_adc1.append(adc1)
                self.win_adc2.append(adc2)
                self.win_v1.append(v1)
                self.win_v2.append(v2)
                self.win_count += 1
                
                # Update progress
                self.progress_bar.setValue(self.win_count)
                self._set_progress_label(self.win_count, self.window_samples)
                
                # Check if window is complete
                if self.win_count >= self.window_samples:
                    self.perform_detection()
                    
                    # Reset window (non-overlapping)
                    self.win_adc1.clear()
                    self.win_adc2.clear()
                    self.win_v1.clear()
                    self.win_v2.clear()
                    self.win_count = 0
                    
        except Exception as e:
            self.log_message(f"Read error: {str(e)}")
            
    def perform_detection(self):
        try:
            # Compute features
            features = compute_features(
                self.win_adc1, self.win_adc2, 
                self.win_v1, self.win_v2
            )
            
            # Create dataframe and align features
            df = pd.DataFrame([features])
            feature_names = self.feature_names or FEATURE_ORDER_FALLBACK
            X = align_features(df, feature_names)
            
            # Perform prediction
            prediction = self.pipeline.predict(X)[0]
            
            # Map prediction to status
            if self.classes is not None:
                try:
                    status = self.classes[int(prediction)]
                except:
                    status = str(prediction)
            else:
                status = str(prediction)
                
            # Map to Indonesian terms
            if status.lower() in ['0', 'healthy', 'sehat']:
                display_status = "Healthy"
            elif status.lower() in ['1', 'mild', 'ringan']:
                display_status = "Mild Infection"  
            elif status.lower() in ['2', 'severe', 'berat']:
                display_status = "Severe Infection"
            else:
                display_status = STATUS_MAPPING.get(status, status)
            
            # Get confidence if available
            confidence = 0.0
            confidence_text = "—"
            if hasattr(self.pipeline, 'predict_proba'):
                try:
                    proba = self.pipeline.predict_proba(X)[0]
                    confidence = float(np.max(proba))
                    confidence_text = f"{confidence:.3f}"
                except:
                    pass
                    
            # Update UI
            self.update_detection_display(display_status, confidence, confidence_text)
            
            # Update statistics
            self.update_detection_stats()
            
            # Log result
            timestamp = self.time_s[-1] if self.time_s else 0.0
            self.log_message(f"Detection @ {timestamp:.1f}s → {display_status} (conf: {confidence_text})")
            
        except Exception as e:
            self.log_message(f"Detection error: {str(e)}")
            
    def update_detection_display(self, status, confidence, confidence_text):
        # Update main status label
        self.status_label.setText(status)
        self.status_label.setStyleSheet(f"""
            color: {STATUS_COLORS.get(status, '#9e9e9e')};
            font-size: 42px;
            font-weight: bold;
            margin: 10px;
        """)
        
        # Update confidence label
        self.confidence_label.setText(f"Confidence: {confidence_text}")
        
        # Update status indicator
        self.status_indicator.set_status(status, confidence)
        
        # Animate color change if needed
        self.animate_status_change(status)
        
    def animate_status_change(self, status):
        # Simple color animation could be added here
        pass
        
    def update_detection_stats(self):
        # Update total detections counter
        current_total = int(self.total_detections.text())
        self.total_detections.setText(str(current_total + 1))
        
        # Calculate and update detection rate
        if hasattr(self, 'first_detection_time'):
            elapsed = time.time() - self.first_detection_time
            rate = (current_total + 1) / (elapsed / 60.0)  # detections per minute
            self.detection_rate.setText(f"{rate:.1f}")
        else:
            self.first_detection_time = time.time()
            self.detection_rate.setText("—")
            
    def update_plot(self):
        if not self.plot_enabled.isChecked() or len(self.time_s) == 0:
            return
            
        # Show only recent data for performance
        keep_samples = int(self.fs * self.plot_win_seconds * 2)
        if len(self.time_s) > keep_samples:
            self.time_s = self.time_s[-keep_samples:]
            self.volt1 = self.volt1[-keep_samples:]
            self.volt2 = self.volt2[-keep_samples:]
            
        # Update curves
        self.curve1.setData(self.time_s, self.volt1)
        self.curve2.setData(self.time_s, self.volt2)
        
        # Set view range to show recent data
        if self.time_s:
            t_end = self.time_s[-1]
            t_start = max(0, t_end - self.plot_win_seconds)
            self.plot_widget.setXRange(t_start, t_end, padding=0.02)
            
    def closeEvent(self, event):
        """Handle application closing"""
        if self.serial:
            self.disconnect_serial()
        event.accept()

def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Ganoderma Radar Detection")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Research Lab")
    
    # Create and show main window
    window = ModernGanodermaGUI()
    window.show()
    
    # Try to load default model if it exists
    default_model = os.path.join(os.path.dirname(__file__), "model_rt.pkl")
    if os.path.exists(default_model):
        window.load_model(default_model)
        window.log_message("Default model loaded automatically")
    
    # Start application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()