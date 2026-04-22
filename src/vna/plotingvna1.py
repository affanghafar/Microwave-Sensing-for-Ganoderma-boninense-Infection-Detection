import sys
import os
import re  # for normalizing feature labels
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QCheckBox, QFileDialog, QPushButton, 
                             QScrollArea, QLabel, QGroupBox, QSizePolicy)

# Mapping of feature units
FEATURE_UNITS = {
    "ε'": "—",       # unitless
    "ε\"": "—",       # unitless
    "tan δ": "—",    # unitless
    "σ": "S/m",      # conductivity
    "Refl.R": "—",
    "Refl.I": "—",
}

class VNAComparativePlotter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VNA Data Comparative Plotter - Side by Side")
        self.setGeometry(100, 100, 1800, 900)  # Increased window height
        
        # Data variables
        self.data = {'Sehat': None, 'Ringan': None, 'Berat': None}
        self.lines = {'Sehat': [], 'Ringan': [], 'Berat': []}
        self.checkboxes = {'Sehat': [], 'Ringan': [], 'Berat': []}
        
        # UI Setup
        self.initUI()
        
    def initUI(self):
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # File buttons
        file_buttons_layout = QHBoxLayout()
        file_buttons_layout.setSpacing(10)
        
        self.file_buttons = {
            'Sehat': QPushButton("Load Data Sehat"),
            'Ringan': QPushButton("Load Data Ringan"),
            'Berat': QPushButton("Load Data Berat")
        }
        
        for condition, button in self.file_buttons.items():
            button.setFixedHeight(40)
            button.clicked.connect(lambda _, c=condition: self.open_file(c))
            file_buttons_layout.addWidget(button)
        
        main_layout.addLayout(file_buttons_layout)
        
        # Plot area - horizontal layout
        plots_layout = QHBoxLayout()
        plots_layout.setSpacing(15)
        main_layout.addLayout(plots_layout)
        
        # Create plot containers for each condition
        self.plot_containers = {}
        self.canvases = {}
        self.scroll_areas = {}
        
        for condition in ['Sehat', 'Ringan', 'Berat']:
            # Create container for each plot
            container = QGroupBox(condition)
            container.setStyleSheet("QGroupBox { font-weight: bold; }")
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(10, 15, 10, 10)
            
            # Matplotlib figure
            fig, ax = plt.subplots(figsize=(6, 4))
            canvas = FigureCanvas(fig)
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.canvases[condition] = (fig, ax, canvas)
            
            # Scroll area for checkboxes
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setMinimumHeight(80)  # Ensure scroll area has enough height
            scroll.setVerticalScrollBarPolicy(1)  # Keep as-is per user's constraint
            
            checkbox_container = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_container)
            checkbox_layout.setSpacing(15)
            checkbox_layout.setContentsMargins(5, 5, 5, 5)
            
            self.scroll_areas[condition] = (scroll, checkbox_container, checkbox_layout)
            
            # Add to container
            container_layout.addWidget(canvas, 4)  # 4 parts for plot
            container_layout.addWidget(scroll, 1)  # 1 part for checkboxes
            
            # Add to main plots layout
            plots_layout.addWidget(container, 1)  # Equal width for each plot
            self.plot_containers[condition] = container
            
        # Status label
        self.status_label = QLabel("Silakan load data untuk masing-masing kondisi")
        self.status_label.setStyleSheet("font-size: 12px; color: #555;")
        main_layout.addWidget(self.status_label)
        
    def open_file(self, condition):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            f"Open Excel File for {condition}", 
            "", 
            "Excel Files (*.xlsx *.xls)"
        )
        
        if file_path:
            try:
                # Read Excel file, skip header row
                self.data[condition] = pd.read_excel(file_path, header=None, skiprows=1)
                
                if len(self.data[condition].columns) >= 2:
                    column_names = ['ε\'', 'ε"', 'σ (S/m)', 'tan(δ)', 'Refl.R', 'Refl.I'][:len(self.data[condition].columns)-1]
                    
                    # Clear previous plot and checkboxes
                    fig, ax, canvas = self.canvases[condition]
                    ax.clear()
                    self.lines[condition] = []
                    self.clear_checkboxes(condition)
                    
                    # Get components
                    scroll, checkbox_container, checkbox_layout = self.scroll_areas[condition]
                    
                    # Get frequency data
                    freq = self.data[condition].iloc[:, 0]
                    
                    # Plot each column and create checkboxes
                    colors = plt.cm.tab10.colors
                    for i, col_name in enumerate(column_names):
                        if (i+1) < len(self.data[condition].columns):
                            # Plot the data
                            line, = ax.plot(freq, self.data[condition].iloc[:, i+1], 
                                          label=col_name, 
                                          color=colors[i % len(colors)],
                                          linewidth=2.5)
                            self.lines[condition].append(line)
                            
                            # Create checkbox
                            checkbox = QCheckBox(col_name)
                            checkbox.setChecked(True)
                            checkbox.stateChanged.connect(lambda _, c=condition: self.update_plot(c))
                            checkbox.setStyleSheet(
                                f"""
                                QCheckBox {{
                                    color: rgb{tuple(int(255*c) for c in colors[i % len(colors)])};
                                    spacing: 5px;
                                    padding: 3px;
                                }}
                                """
                            )
                            self.checkboxes[condition].append(checkbox)
                            checkbox_layout.addWidget(checkbox)
                    
                    # Set plot properties (bigger & bold axis labels)
                    ax.set_xlabel('Frequency (MHz)', fontsize=10, fontweight='bold')
                    ax.set_ylabel('Value', fontsize=10, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    # initial legend will be replaced by _update_legend()
                    ax.legend(loc='upper right', fontsize=8)
                    ax.tick_params(axis='both', labelsize=12)
                    # Leave room on the left and top to avoid clipping
                    fig.subplots_adjust(left=0.13, bottom=0.10, right=0.995, top=0.94)
                    
                    # Update scroll area
                    scroll.setWidget(checkbox_container)
                    
                    # Update status
                    self.status_label.setText(f"Data {condition} berhasil dimuat - {file_path}")
                    
                    # --- Title from filename prefix (akar_label_index, pelepah_*, daun_*) ---
                    title = self._infer_bagian_title(file_path)
                    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.975)
                    fig.subplots_adjust(left=0.13, bottom=0.10, right=0.995, top=0.94)
                    # Re-apply layout with safe margins
                    fig.subplots_adjust(left=0.13, bottom=0.10, right=0.995, top=0.94)
                    
                    # Update y-axis label, limits, and legend
                    self._update_ylabel(condition)
                    self._update_ylim(condition)
                    self._update_legend(condition)
                    # Redraw canvas
                    canvas.draw()
                    
            except Exception as e:
                self.status_label.setText(f"Error loading {condition} data: {str(e)}")
                print(f"Error reading {condition} file: {e}")
    
    def clear_checkboxes(self, condition):
        _, _, checkbox_layout = self.scroll_areas[condition]
        for i in reversed(range(checkbox_layout.count())):
            widget = checkbox_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        self.checkboxes[condition] = []

    # Normalize feature label text into canonical base names
    def _canonical_feature_name(self, text: str) -> str:
        """
        Remove any trailing parentheses and normalize to canonical labels:
        - "ε'" -> "ε'",  "ε\"" -> "ε\""
        - "tan(δ)" -> "tan δ"
        - "σ (S/m)" -> "σ"
        """
        base = re.sub(r"\s*\(.*?\)\s*$", "", text).strip()
        base = base.replace("ε'", "ε'").replace('ε"', 'ε"').replace("tan(δ)", "tan δ")
        return base

    # Infer plot title (bagian) from filename prefix: akar_label_index, pelepah_*, daun_*
    def _infer_bagian_title(self, file_path: str) -> str:
        name = os.path.basename(file_path).lower()
        stem = os.path.splitext(name)[0]
        tokens = [t for t in stem.split('_') if t]

        bagian_map = {'akar': 'Akar', 'pelepah': 'Pelepah', 'daun': 'Daun'}
        label_map  = {'sehat': 'Sehat', 'ringan': 'Ringan', 'berat': 'Berat'}

        parts = []
        if tokens:
            bagian = bagian_map.get(tokens[0])
            if bagian:
                parts.append(bagian)
        if len(tokens) >= 2:
            label = label_map.get(tokens[1])
            if label:
                parts.append(label)

        return ' '.join(parts) if parts else 'Data'

    # Update Y-axis label based on visible features
    def _update_ylabel(self, condition: str) -> None:
        fig, ax, canvas = self.canvases[condition]
        visible_features = []
        for cb in self.checkboxes[condition]:
            if cb.isChecked():
                visible_features.append(self._canonical_feature_name(cb.text()))
        if not visible_features:
            ax.set_ylabel("Value", fontsize=10, fontweight='bold')
            return
        units = {FEATURE_UNITS.get(name, "—") for name in visible_features}
        if len(visible_features) == 1:
            name = visible_features[0]
            unit = next(iter(units))
            ax.set_ylabel(f"{name} ({unit})", fontsize=10, fontweight='bold')
        elif len(units) == 1:
            unit = next(iter(units))
            ax.set_ylabel(f"Value ({unit})", fontsize=10, fontweight='bold')
        else:
            ax.set_ylabel("Value (mixed units)", fontsize=10, fontweight='bold')

    # Update Y-axis limits adaptively based on visible lines (both min and max) with proportional padding
    def _update_ylim(self, condition: str) -> None:
        fig, ax, canvas = self.canvases[condition]
        y_max = None
        y_min = None
        for cb, line in zip(self.checkboxes[condition], self.lines[condition]):
            if cb.isChecked() and line is not None:
                y = line.get_ydata()
                if y is None or len(y) == 0:
                    continue
                try:
                    cur_max = np.nanmax(y)
                    cur_min = np.nanmin(y)
                except Exception:
                    continue
                if y_max is None or cur_max > y_max:
                    y_max = cur_max
                if y_min is None or cur_min < y_min:
                    y_min = cur_min
        if y_max is None or y_min is None:
            ax.autoscale()
            return
        span = y_max - y_min
        # Use 5% of span as padding; if span ~ 0, use a tiny epsilon
        pad = 0.05 * span if np.isfinite(span) and span > 0 else 1e-9
        lower = y_min - pad
        upper = y_max + pad
        if not np.isfinite(upper) or not np.isfinite(lower) or upper <= lower:
            ax.autoscale()
            return
        ax.set_ylim(lower, upper)

    # Update legend to show only visible features using canonical labels
    def _update_legend(self, condition: str) -> None:
        fig, ax, canvas = self.canvases[condition]
        handles = []
        labels = []
        for cb, line in zip(self.checkboxes[condition], self.lines[condition]):
            if line is None:
                continue
            visible = cb.isChecked() and line.get_visible()
            if visible:
                handles.append(line)
                labels.append(self._canonical_feature_name(cb.text()))
        if handles:
            ax.legend(handles, labels, loc='upper right', fontsize=8)
        else:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
    
    def update_plot(self, condition):
        fig, ax, canvas = self.canvases[condition]
        for checkbox, line in zip(self.checkboxes[condition], self.lines[condition]):
            line.set_visible(checkbox.isChecked())
        # Update y-axis label, limits, and legend after toggling
        self._update_ylabel(condition)
        self._update_ylim(condition)
        self._update_legend(condition)
        canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VNAComparativePlotter()
    window.show()
    sys.exit(app.exec_())