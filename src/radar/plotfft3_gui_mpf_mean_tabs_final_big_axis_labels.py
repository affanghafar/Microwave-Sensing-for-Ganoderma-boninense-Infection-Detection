
import os
import re
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.ttk as ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Gunakan backend TkAgg untuk embed di Tkinter
plt.switch_backend("TkAgg")


def moving_average(x, w):
    """Simple moving average dengan window w."""
    x = np.asarray(x, dtype=float)
    if w <= 1 or len(x) < w:
        return x.copy()
    return np.convolve(x, np.ones(w, dtype=float) / float(w), mode="valid")


class MPFMeanPowerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MPF & Mean Power GUI (Tabs, Drop per Tab, Trend, Range, Auto Sample Size)")

        # --- data mentah ---
        self.raw_time = None
        # simpan terpisah: tegangan vs ADC (counts)
        self.raw_v1_voltage = None
        self.raw_v2_voltage = None
        self.raw_v1_adc = None
        self.raw_v2_adc = None
        # sinyal yang sedang dipakai untuk pemrosesan (mengikuti mode)
        self.raw_v1 = None
        self.raw_v2 = None

        # --- hasil pemrosesan (berbasis MA1) ---
        self.time_labels_export = None  # label HH:MM per titik MA1
        self.mpf_adc1_ma1 = None
        self.mpf_adc2_ma1 = None
        self.mpf_adc1_ma2 = None
        self.mpf_adc2_ma2 = None
        self.mean_power_adc1_ma1 = None
        self.mean_power_adc2_ma1 = None
        self.mean_power_adc1_ma2 = None
        self.mean_power_adc2_ma2 = None

        # --- drop mask per tab (boolean di domain time_labels_export) ---
        self.drop_mask_mpf = None
        self.drop_mask_mean = None

        # --- parameter MA (bisa diganti via GUI) ---
        self.ma1_var = tk.StringVar(value="19")
        self.ma2_var = tk.StringVar(value="59")

        # --- panjang chunk FFT (sample_size), auto dari nama file *_512_* / *_1024_* ---
        self.sample_size = 512

        # --- mode data: 'voltage' atau 'adc' ---
        self.data_mode_var = tk.StringVar(value="voltage")

        # --- visibility (berlaku untuk kedua tab) ---
        self.show_ma1_var = tk.BooleanVar(value=True)
        self.show_ma2_var = tk.BooleanVar(value=True)
        self.show_trend_var = tk.BooleanVar(value=True)

        # --- kontrol drop per tab (format HH:MM) ---
        self.drop_from_mpf_var = tk.StringVar(value="")
        self.drop_to_mpf_var = tk.StringVar(value="")
        self.drop_from_mean_var = tk.StringVar(value="")
        self.drop_to_mean_var = tk.StringVar(value="")

        # --- info range Mean Power (4 nilai) ---
        self.range_adc1_ma1_var = tk.StringVar(value="Batang Bawah MA1 range: -")
        self.range_adc1_ma2_var = tk.StringVar(value="Batang Bawah MA2 range: -")
        self.range_adc2_ma1_var = tk.StringVar(value="Batang Atas MA1 range: -")
        self.range_adc2_ma2_var = tk.StringVar(value="Batang Atas MA2 range: -")

        self._build_ui()

    # ============================================================
    # UI
    # ============================================================
    def _build_ui(self):
        # Baris info range di atas tombol
        range_bar = tk.Frame(self)
        range_bar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(3, 0))
        tk.Label(range_bar, text="Mean Power Ranges (dB):",
                 font=("TkDefaultFont", 9, "bold")).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(range_bar, textvariable=self.range_adc1_ma1_var).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(range_bar, textvariable=self.range_adc1_ma2_var).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(range_bar, textvariable=self.range_adc2_ma1_var).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(range_bar, textvariable=self.range_adc2_ma2_var).pack(side=tk.LEFT, padx=(0, 8))

        top = tk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # --- tombol file & data ---
        tk.Button(top, text="Input Data (Excel)", command=self.load_data).pack(side=tk.LEFT, padx=5)
        tk.Button(top, text="Save Data (.xlsx)", command=self.save_data).pack(side=tk.LEFT, padx=5)

        # --- MA controls ---
        ma_frame = tk.Frame(top)
        ma_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(ma_frame, text="MA1:").pack(side=tk.LEFT)
        tk.Entry(ma_frame, width=4, textvariable=self.ma1_var).pack(side=tk.LEFT, padx=(0, 5))
        tk.Label(ma_frame, text="MA2:").pack(side=tk.LEFT)
        tk.Entry(ma_frame, width=4, textvariable=self.ma2_var).pack(side=tk.LEFT)
        tk.Button(top, text="Apply MA", command=self.apply_ma).pack(side=tk.LEFT, padx=5)

        # --- pilihan mode data: tegangan vs ADC ---
        mode_frame = tk.Frame(top)
        mode_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(mode_frame, text="Data:").pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="Tegangan (V)", variable=self.data_mode_var,
                       value="voltage", command=self.on_change_mode).pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="ADC", variable=self.data_mode_var,
                       value="adc", command=self.on_change_mode).pack(side=tk.LEFT, padx=(5, 0))

        # --- checkbox show/hide ---
        cb_frame = tk.Frame(top)
        cb_frame.pack(side=tk.LEFT, padx=10)
        tk.Checkbutton(cb_frame, text="MA1", variable=self.show_ma1_var,
                       command=self.update_all_plots).pack(side=tk.LEFT)
        tk.Checkbutton(cb_frame, text="MA2", variable=self.show_ma2_var,
                       command=self.update_all_plots).pack(side=tk.LEFT, padx=(5, 0))
        tk.Checkbutton(cb_frame, text="Trend", variable=self.show_trend_var,
                       command=self.update_all_plots).pack(side=tk.LEFT, padx=(5, 0))

        # --- drop MPF ---
        drop_mpf_frame = tk.Frame(top)
        drop_mpf_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(drop_mpf_frame, text="Drop MPF (HH:MM) from").pack(side=tk.LEFT)
        tk.Entry(drop_mpf_frame, width=6, textvariable=self.drop_from_mpf_var).pack(side=tk.LEFT, padx=(0, 3))
        tk.Label(drop_mpf_frame, text="to").pack(side=tk.LEFT)
        tk.Entry(drop_mpf_frame, width=6, textvariable=self.drop_to_mpf_var).pack(side=tk.LEFT, padx=(0, 3))
        tk.Button(drop_mpf_frame, text="Apply", command=self.apply_drop_mpf).pack(side=tk.LEFT)

        # --- drop Mean Power ---
        drop_mean_frame = tk.Frame(top)
        drop_mean_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(drop_mean_frame, text="Drop Mean (HH:MM) from").pack(side=tk.LEFT)
        tk.Entry(drop_mean_frame, width=6, textvariable=self.drop_from_mean_var).pack(side=tk.LEFT, padx=(0, 3))
        tk.Label(drop_mean_frame, text="to").pack(side=tk.LEFT)
        tk.Entry(drop_mean_frame, width=6, textvariable=self.drop_to_mean_var).pack(side=tk.LEFT, padx=(0, 3))
        tk.Button(drop_mean_frame, text="Apply", command=self.apply_drop_mean).pack(side=tk.LEFT)

        # --- save images ---
        tk.Button(top, text="Save MPF Image", command=self.save_mpf_image).pack(side=tk.LEFT, padx=5)
        tk.Button(top, text="Save Mean Image", command=self.save_mean_image).pack(side=tk.LEFT, padx=5)

        # --- reset grafik (tanpa hapus batas waktu) ---
        tk.Button(top, text="Reset Graph", command=self.reset_graphs).pack(side=tk.LEFT, padx=5)

        # --- Quit ---
        tk.Button(top, text="Quit", command=self.destroy).pack(side=tk.RIGHT, padx=5)

        # --- Notebook Tabs ---
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab MPF
        frame_mpf = tk.Frame(notebook)
        notebook.add(frame_mpf, text="MPF")
        self.fig_mpf, (self.ax_mpf1, self.ax_mpf2) = plt.subplots(2, 1, figsize=(8, 6))
        self.canvas_mpf = FigureCanvasTkAgg(self.fig_mpf, master=frame_mpf)
        self.canvas_mpf.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.fig_mpf.subplots_adjust(left=0.1, right=0.95, top=0.93, bottom=0.12, hspace=0.35)

        # Tab Mean Power
        frame_mean = tk.Frame(notebook)
        notebook.add(frame_mean, text="Mean Power")
        self.fig_mean, (self.ax_mean1, self.ax_mean2) = plt.subplots(2, 1, figsize=(8, 6))
        self.canvas_mean = FigureCanvasTkAgg(self.fig_mean, master=frame_mean)
        self.canvas_mean.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.fig_mean.subplots_adjust(left=0.1, right=0.95, top=0.93, bottom=0.12, hspace=0.35)

    # ============================================================
    # DETEKSI KOLOM
    # ============================================================
    def _detect_columns(self, df: pd.DataFrame):
        cols = list(df.columns)

        # waktu
        time_col = None
        preferred_time = ["Time (s) - COM11", "Time (s)", "time", "Time"]
        for cand in preferred_time:
            if cand in cols:
                time_col = cand
                break
        if time_col is None:
            for col in cols:
                if "time" in str(col).lower():
                    time_col = col
                    break
        if time_col is None:
            raise KeyError("Kolom waktu tidak ditemukan. Pastikan ada kolom 'Time (s)' atau sejenisnya.")

        # --- tegangan ADC1/ADC2 (Voltage) ---
        v1_volt_col = None
        v2_volt_col = None

        preferred_v1_volt = ["Voltage ADC1 (V) - COM11", "Voltage ADC1 (V)"]
        preferred_v2_volt = ["Voltage ADC2 (V) - COM11", "Voltage ADC2 (V)"]
        for cand in preferred_v1_volt:
            if cand in cols:
                v1_volt_col = cand
                break
        for cand in preferred_v2_volt:
            if cand in cols:
                v2_volt_col = cand
                break

        if v1_volt_col is None:
            for col in cols:
                low = str(col).lower()
                if ("volt" in low or "(v" in low) and ("adc1" in low or "ch1" in low or "channel 1" in low):
                    v1_volt_col = col
                    break
        if v2_volt_col is None:
            for col in cols:
                low = str(col).lower()
                if ("volt" in low or "(v" in low) and ("adc2" in low or "ch2" in low or "channel 2" in low):
                    v2_volt_col = col
                    break

        # --- nilai ADC (counts) ADC1/ADC2 ---
        v1_adc_col = None
        v2_adc_col = None

        preferred_v1_adc = ["ADC1", "ADC 1", "ADC1 (counts)", "ADC 1 (counts)", "ADC 1(16-bit)"]
        preferred_v2_adc = ["ADC2", "ADC 2", "ADC2 (counts)", "ADC 2 (counts)", "ADC 2(16-bit)"]
        for cand in preferred_v1_adc:
            if cand in cols:
                v1_adc_col = cand
                break
        for cand in preferred_v2_adc:
            if cand in cols:
                v2_adc_col = cand
                break

        if v1_adc_col is None:
            for col in cols:
                low = str(col).lower()
                if ("adc1" in low or "ch1" in low or "channel 1" in low) and ("volt" not in low) and "(v" not in low:
                    v1_adc_col = col
                    break
        if v2_adc_col is None:
            for col in cols:
                low = str(col).lower()
                if ("adc2" in low or "ch2" in low or "channel 2" in low) and ("volt" not in low) and "(v" not in low:
                    v2_adc_col = col
                    break

        if v1_volt_col is None and v1_adc_col is None:
            raise KeyError("Kolom sinyal ADC1 (tegangan atau ADC) tidak ditemukan.")
        if v2_volt_col is None and v2_adc_col is None:
            raise KeyError("Kolom sinyal ADC2 (tegangan atau ADC) tidak ditemukan.")

        time = df[time_col].to_numpy(dtype=float)

        v1_voltage = df[v1_volt_col].to_numpy(dtype=float) if v1_volt_col is not None else None
        v2_voltage = df[v2_volt_col].to_numpy(dtype=float) if v2_volt_col is not None else None
        v1_adc = df[v1_adc_col].to_numpy(dtype=float) if v1_adc_col is not None else None
        v2_adc = df[v2_adc_col].to_numpy(dtype=float) if v2_adc_col is not None else None

        return {
            "time": time,
            "v1_voltage": v1_voltage,
            "v2_voltage": v2_voltage,
            "v1_adc": v1_adc,
            "v2_adc": v2_adc,
        }

    def _update_raw_signals_from_mode(self, show_message: bool = True) -> bool:
        """Pilih sinyal mentah (tegangan / ADC) sesuai pilihan GUI."""
        if self.raw_time is None:
            return False

        mode = self.data_mode_var.get()
        if mode == "adc":
            if self.raw_v1_adc is not None and self.raw_v2_adc is not None:
                self.raw_v1 = self.raw_v1_adc
                self.raw_v2 = self.raw_v2_adc
                return True
            else:
                if show_message:
                    messagebox.showwarning(
                        "Data ADC tidak tersedia",
                        "File ini tidak memiliki kolom ADC yang terdeteksi. Menggunakan tegangan (V).",
                    )
                self.data_mode_var.set("voltage")
                mode = "voltage"

        if mode == "voltage":
            if self.raw_v1_voltage is not None and self.raw_v2_voltage is not None:
                self.raw_v1 = self.raw_v1_voltage
                self.raw_v2 = self.raw_v2_voltage
                return True
            else:
                if show_message:
                    messagebox.showerror(
                        "Error",
                        "Kolom tegangan (Voltage) tidak ditemukan di file ini."
                    )
                return False

        return False

    # ============================================================
    # CORE PROCESSING (MPF & MEAN POWER)
    # ============================================================
    def _process_signal(self, time, v1, v2, sample_size=512, w1=19, w2=59):
        time = np.asarray(time, dtype=float)
        v1 = np.asarray(v1, dtype=float)
        v2 = np.asarray(v2, dtype=float)

        if len(time) < 2:
            raise ValueError("Data waktu terlalu pendek.")
        dt = time[1] - time[0]
        if dt <= 0:
            raise ValueError("Time step tidak valid (dt <= 0). Pastikan waktu naik dan teratur.")
        fs = 1.0 / dt

        n_samples = len(v1)
        n_chunks = (n_samples // sample_size) + (1 if n_samples % sample_size else 0)

        mpf_adc1 = []
        mpf_adc2 = []
        mean_power_adc1 = []
        mean_power_adc2 = []

        window = np.hanning(sample_size)

        for i in range(n_chunks):
            start_idx = i * sample_size
            end_idx = min(start_idx + sample_size, n_samples)

            seg1 = v1[start_idx:end_idx]
            seg2 = v2[start_idx:end_idx]

            # pad dengan nol jika chunk terakhir lebih pendek
            if len(seg1) < sample_size:
                seg1 = np.pad(seg1, (0, sample_size - len(seg1)), mode="constant")
                seg2 = np.pad(seg2, (0, sample_size - len(seg2)), mode="constant")

            # hilangkan DC
            seg1 = seg1 - np.mean(seg1)
            seg2 = seg2 - np.mean(seg2)

            # windowing
            seg1w = seg1 * window
            seg2w = seg2 * window

            # FFT (dinormalisasi sample_size)
            fft_adc1 = fft(seg1w) / sample_size
            fft_adc2 = fft(seg2w) / sample_size

            freqs = fftfreq(sample_size, d=1.0 / fs)
            pos_mask = freqs >= 0
            freqs_pos = freqs[pos_mask]

            power_adc1 = np.abs(fft_adc1[pos_mask]) ** 2
            power_adc2 = np.abs(fft_adc2[pos_mask]) ** 2

            power_adc1_db = 10 * np.log10(power_adc1 + 1e-10)
            power_adc2_db = 10 * np.log10(power_adc2 + 1e-10)

            # --- MPF ---
            power_adc1_linear = 10 ** (power_adc1_db / 10.0)
            power_adc2_linear = 10 ** (power_adc2_db / 10.0)

            s1 = power_adc1_linear.sum()
            s2 = power_adc2_linear.sum()
            power_adc1_norm = power_adc1_linear / s1 if s1 != 0 else power_adc1_linear
            power_adc2_norm = power_adc2_linear / s2 if s2 != 0 else power_adc2_linear

            mpf1 = float(np.sum(freqs_pos * power_adc1_norm)) if power_adc1_norm.sum() != 0 else 0.0
            mpf2 = float(np.sum(freqs_pos * power_adc2_norm)) if power_adc2_norm.sum() != 0 else 0.0
            mpf_adc1.append(mpf1)
            mpf_adc2.append(mpf2)

            # --- Mean Power (dB) ---
            mean_power_adc1.append(float(np.mean(power_adc1_db)) if len(power_adc1_db) > 0 else 0.0)
            mean_power_adc2.append(float(np.mean(power_adc2_db)) if len(power_adc2_db) > 0 else 0.0)

        mpf_adc1 = np.asarray(mpf_adc1, dtype=float)
        mpf_adc2 = np.asarray(mpf_adc2, dtype=float)
        mean_power_adc1 = np.asarray(mean_power_adc1, dtype=float)
        mean_power_adc2 = np.asarray(mean_power_adc2, dtype=float)

        # supaya grafik belakang konsisten: buang 1 titik terakhir (sesuai script awal)
        if len(mpf_adc1) > 1:
            plot_len = len(mpf_adc1) - 1
            mpf_adc1 = mpf_adc1[:plot_len]
            mpf_adc2 = mpf_adc2[:plot_len]
            mean_power_adc1 = mean_power_adc1[:plot_len]
            mean_power_adc2 = mean_power_adc2[:plot_len]
        else:
            plot_len = len(mpf_adc1)

        # label waktu per chunk (serupa script awal, 1 menit per chunk mulai 08:30)
        def get_time_labels(n_chunks, start_hour=8, start_minute=0, chunk_minutes=1):
            labels = []
            start = datetime.datetime(2025, 6, 19, start_hour, start_minute)
            for i in range(n_chunks):
                t = start + datetime.timedelta(minutes=i * chunk_minutes)
                labels.append(t.strftime("%H:%M"))
            return labels

        all_labels = np.array(get_time_labels(n_chunks, start_hour=8, start_minute=0, chunk_minutes=1), dtype=str)
        base_labels = all_labels[:plot_len]

        # --- Moving average ---
        w1 = int(w1)
        w2 = int(w2)

        mpf_adc1_ma1 = moving_average(mpf_adc1, w1)
        mpf_adc2_ma1 = moving_average(mpf_adc2, w1)
        mpf_adc1_ma2 = moving_average(mpf_adc1, w2)
        mpf_adc2_ma2 = moving_average(mpf_adc2, w2)

        mean_power_adc1_ma1 = moving_average(mean_power_adc1, w1)
        mean_power_adc2_ma1 = moving_average(mean_power_adc2, w1)
        mean_power_adc1_ma2 = moving_average(mean_power_adc1, w2)
        mean_power_adc2_ma2 = moving_average(mean_power_adc2, w2)

        # label waktu untuk MA1 (centered)
        if plot_len > w1:
            lbl_ma1 = base_labels[(w1 - 1) // 2: -(w1 // 2)]
        else:
            lbl_ma1 = base_labels

        L = len(lbl_ma1)
        mpf_adc1_ma1 = np.asarray(mpf_adc1_ma1[:L], dtype=float)
        mpf_adc2_ma1 = np.asarray(mpf_adc2_ma1[:L], dtype=float)
        mean_power_adc1_ma1 = np.asarray(mean_power_adc1_ma1[:L], dtype=float)
        mean_power_adc2_ma1 = np.asarray(mean_power_adc2_ma1[:L], dtype=float)

        # MA2 boleh beda panjang, nanti akan di-interp pada saat plotting
        mpf_adc1_ma2 = np.asarray(mpf_adc1_ma2, dtype=float)
        mpf_adc2_ma2 = np.asarray(mpf_adc2_ma2, dtype=float)
        mean_power_adc1_ma2 = np.asarray(mean_power_adc1_ma2, dtype=float)
        mean_power_adc2_ma2 = np.asarray(mean_power_adc2_ma2, dtype=float)

        return {
            "time_labels_ma1": np.array(lbl_ma1, dtype=str),
            "mpf_adc1_ma1": mpf_adc1_ma1,
            "mpf_adc2_ma1": mpf_adc2_ma1,
            "mpf_adc1_ma2": mpf_adc1_ma2,
            "mpf_adc2_ma2": mpf_adc2_ma2,
            "mean_power_adc1_ma1": mean_power_adc1_ma1,
            "mean_power_adc2_ma1": mean_power_adc2_ma1,
            "mean_power_adc1_ma2": mean_power_adc1_ma2,
            "mean_power_adc2_ma2": mean_power_adc2_ma2,
        }

    # ============================================================
    # LOAD DATA & APPLY MA
    # ============================================================
    def load_data(self):
        path = filedialog.askopenfilename(
            title="Pilih file Excel",
            filetypes=[("Excel file", "*.xlsx *.xls")]
        )
        if not path:
            return

        # --- deteksi sample_size dari nama file, misal *_512_*, *_1024_* ---
        base = os.path.basename(path)
        m = re.search(r"_(\d+)_", base)
        if m:
            val = int(m.group(1))
            if val in (512, 1024):
                self.sample_size = val
            else:
                self.sample_size = 512
        else:
            self.sample_size = 512

        try:
            df = pd.read_excel(path)
        except Exception as e:
            messagebox.showerror("Error", f"Gagal membaca file Excel:\n{e}")
            return

        try:
            cols = self._detect_columns(df)
        except KeyError as e:
            messagebox.showerror("Error", str(e))
            return

        self.raw_time = cols["time"]
        self.raw_v1_voltage = cols["v1_voltage"]
        self.raw_v2_voltage = cols["v2_voltage"]
        self.raw_v1_adc = cols["v1_adc"]
        self.raw_v2_adc = cols["v2_adc"]

        # pilih sinyal sesuai mode (default: tegangan)
        if not self._update_raw_signals_from_mode(show_message=True):
            return

        # reset drop masks
        self.drop_mask_mpf = None
        self.drop_mask_mean = None

        # proses awal dengan MA default
        self.apply_ma(initial=True)

    def on_change_mode(self):
        """Dipanggil saat radiobutton Tegangan/ADC diganti."""
        if self.raw_time is None:
            return
        if not self._update_raw_signals_from_mode(show_message=True):
            return
        # setelah ganti mode data, proses ulang dengan MA yang sama
        self.apply_ma(initial=True)

    def apply_ma(self, initial=False):
        if self.raw_time is None or self.raw_v1 is None or self.raw_v2 is None:
            if not initial:
                messagebox.showwarning("Belum ada data", "Silakan Input Data (Excel) terlebih dahulu.")
            return

        # reset drop setiap kali MA diganti
        self.drop_mask_mpf = None
        self.drop_mask_mean = None

        try:
            ma1 = int(self.ma1_var.get())
            ma2 = int(self.ma2_var.get())
            if ma1 < 1 or ma2 < 1:
                raise ValueError("MA harus >= 1")
        except Exception as e:
            messagebox.showerror("Error", f"Nilai MA tidak valid:\n{e}")
            return

        try:
            result = self._process_signal(
                self.raw_time,
                self.raw_v1,
                self.raw_v2,
                sample_size=self.sample_size,
                w1=ma1,
                w2=ma2,
            )
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memproses data:\n{e}")
            return

        self._store_result(result)
        self.update_all_plots()

    def _store_result(self, result: dict):
        self.time_labels_export = result["time_labels_ma1"]
        self.mpf_adc1_ma1 = result["mpf_adc1_ma1"]
        self.mpf_adc2_ma1 = result["mpf_adc2_ma1"]
        self.mpf_adc1_ma2 = result["mpf_adc1_ma2"]
        self.mpf_adc2_ma2 = result["mpf_adc2_ma2"]
        self.mean_power_adc1_ma1 = result["mean_power_adc1_ma1"]
        self.mean_power_adc2_ma1 = result["mean_power_adc2_ma1"]
        self.mean_power_adc1_ma2 = result["mean_power_adc1_ma2"]
        self.mean_power_adc2_ma2 = result["mean_power_adc2_ma2"]

        # reset info range
        self.range_adc1_ma1_var.set("Batang Bawah MA1 range: -")
        self.range_adc1_ma2_var.set("Batang Bawah MA2 range: -")
        self.range_adc2_ma1_var.set("Batang Atas MA1 range: -")
        self.range_adc2_ma2_var.set("Batang Atas MA2 range: -")

    # ============================================================
    # DROP INTERVAL PER TAB
    # ============================================================
    def _valid_hhmm(self, s: str) -> bool:
        try:
            parts = s.split(":")
            if len(parts) != 2:
                return False
            h = int(parts[0])
            m = int(parts[1])
            return 0 <= h <= 23 and 0 <= m <= 59
        except Exception:
            return False

    def apply_drop_mpf(self):
        self._apply_drop_generic("mpf", self.drop_from_mpf_var, self.drop_to_mpf_var)

    def apply_drop_mean(self):
        self._apply_drop_generic("mean", self.drop_from_mean_var, self.drop_to_mean_var)

    def _apply_drop_generic(self, tab: str, from_var: tk.StringVar, to_var: tk.StringVar):
        if self.time_labels_export is None:
            messagebox.showwarning("Belum ada data", "Silakan Input Data (Excel) dan Apply MA terlebih dahulu.")
            return

        start_label = from_var.get().strip()
        end_label = to_var.get().strip()

        if not start_label or not end_label:
            messagebox.showerror("Error", "Isi kedua kolom 'from' dan 'to' (format HH:MM).")
            return

        if not (self._valid_hhmm(start_label) and self._valid_hhmm(end_label)):
            messagebox.showerror("Error", "Format waktu harus HH:MM, misalnya 08:30.")
            return

        labels = np.array(self.time_labels_export, dtype=str)
        idx_to_drop = [i for i, lab in enumerate(labels) if start_label <= lab <= end_label]
        if not idx_to_drop:
            messagebox.showwarning("Tidak ada yang dibuang", "Tidak ada titik waktu di rentang tersebut.")
            return

        if tab == "mpf":
            if self.drop_mask_mpf is None or len(self.drop_mask_mpf) != len(labels):
                self.drop_mask_mpf = np.ones(len(labels), dtype=bool)
            for i in idx_to_drop:
                self.drop_mask_mpf[i] = False
        elif tab == "mean":
            if self.drop_mask_mean is None or len(self.drop_mask_mean) != len(labels):
                self.drop_mask_mean = np.ones(len(labels), dtype=bool)
            for i in idx_to_drop:
                self.drop_mask_mean[i] = False

        self.update_all_plots()

    def reset_graphs(self):
        """Reset semua drop di kedua tab dan kembalikan grafik ke kondisi penuh,
        tanpa menghapus batas waktu yang sudah diisi di GUI."""
        self.drop_mask_mpf = None
        self.drop_mask_mean = None
        self.update_all_plots()

    # ============================================================
    # PLOT UPDATERS
    # ============================================================
    def update_all_plots(self):
        self.update_mpf_plot()
        self.update_mean_power_plot()

    def update_mpf_plot(self):
        if self.time_labels_export is None:
            return

        L = len(self.time_labels_export)
        if L == 0:
            return

        try:
            ma1 = int(self.ma1_var.get())
        except Exception:
            ma1 = None
        try:
            ma2 = int(self.ma2_var.get())
        except Exception:
            ma2 = None

        label_ma1 = f"MA{ma1}" if ma1 and ma1 > 0 else "MA1"
        label_ma2 = f"MA{ma2}" if ma2 and ma2 > 0 else "MA2"

        show_ma1 = self.show_ma1_var.get()
        show_ma2 = self.show_ma2_var.get()
        show_trend = self.show_trend_var.get()

        # mask drop MPF
        if self.drop_mask_mpf is not None and len(self.drop_mask_mpf) == L:
            mask = self.drop_mask_mpf
        else:
            mask = np.ones(L, dtype=bool)

        kept_idx = np.where(mask)[0]
        if kept_idx.size == 0:
            self.ax_mpf1.clear()
            self.ax_mpf2.clear()
            self.ax_mpf1.set_title("MPF (no data after drop)")
            self.ax_mpf2.set_title("MPF (no data after drop)")
            self.canvas_mpf.draw()
            return

        x = np.arange(kept_idx.size)
        labels_comp = np.array(self.time_labels_export)[kept_idx]

        self.ax_mpf1.clear()
        self.ax_mpf2.clear()

        # ---------- Batang Atas (ADC2) ----------
        if show_ma1 and self.mpf_adc2_ma1 is not None:
            y_top_full = np.asarray(self.mpf_adc2_ma1, dtype=float)
            y_top = y_top_full[kept_idx]
            self.ax_mpf1.plot(x, y_top, label=f"MPF Batang Atas ({label_ma1})", color="orange", linewidth=2, alpha=0.8)

        if show_ma2 and self.mpf_adc2_ma2 is not None and self.mpf_adc2_ma2.size > 0:
            M_top = self.mpf_adc2_ma2.size
            x2_base = np.linspace(0, L - 1, num=M_top)
            y2_top_full = np.asarray(self.mpf_adc2_ma2, dtype=float)
            y2_top = np.interp(kept_idx.astype(float), x2_base, y2_top_full)
            self.ax_mpf1.plot(x, y2_top, label=f"MPF Batang Atas ({label_ma2})", color="green", linewidth=2, alpha=0.8)

        if show_trend and self.mpf_adc2_ma1 is not None:
            y_tr_top = np.asarray(self.mpf_adc2_ma1, dtype=float)[kept_idx]
            if y_tr_top.size >= 2:
                coef_top = np.polyfit(x, y_tr_top, 1)
                t_top = np.poly1d(coef_top)(x)
                self.ax_mpf1.plot(x, t_top, label="Trend Batang Atas", color="black",
                                   linestyle="--", linewidth=2, alpha=0.8)

        self.ax_mpf1.set_title("Mean Power Frequency (MPF) Trend (Batang Atas)", fontsize=22, fontweight="bold")
        self.ax_mpf1.set_xlabel("Time", fontsize=22, fontweight="bold")
        self.ax_mpf1.set_ylabel("Frequency (Hz)", fontsize=22, fontweight="bold")
        self.ax_mpf1.grid(True, linestyle="--", alpha=0.7)
        if show_ma1 or show_ma2 or show_trend:
            self.ax_mpf1.legend(fontsize=9)

        # ---------- Batang Bawah (ADC1) ----------
        if show_ma1 and self.mpf_adc1_ma1 is not None:
            y_bot_full = np.asarray(self.mpf_adc1_ma1, dtype=float)
            y_bot = y_bot_full[kept_idx]
            self.ax_mpf2.plot(x, y_bot, label=f"MPF Batang Bawah ({label_ma1})", color="blue", linewidth=2, alpha=0.8)

        if show_ma2 and self.mpf_adc1_ma2 is not None and self.mpf_adc1_ma2.size > 0:
            M_bot = self.mpf_adc1_ma2.size
            x2b_base = np.linspace(0, L - 1, num=M_bot)
            y2_bot_full = np.asarray(self.mpf_adc1_ma2, dtype=float)
            y2_bot = np.interp(kept_idx.astype(float), x2b_base, y2_bot_full)
            self.ax_mpf2.plot(x, y2_bot, label=f"MPF Batang Bawah ({label_ma2})", color="green", linewidth=2, alpha=0.8)

        if show_trend and self.mpf_adc1_ma1 is not None:
            y_tr_bot = np.asarray(self.mpf_adc1_ma1, dtype=float)[kept_idx]
            if y_tr_bot.size >= 2:
                coef_bot = np.polyfit(x, y_tr_bot, 1)
                t_bot = np.poly1d(coef_bot)(x)
                self.ax_mpf2.plot(x, t_bot, label="Trend Batang Bawah", color="black",
                                   linestyle="--", linewidth=2, alpha=0.8)

        self.ax_mpf2.set_title("Mean Power Frequency (MPF) Trend (Batang Bawah)", fontsize=22, fontweight="bold")
        self.ax_mpf2.set_xlabel("Time", fontsize=22, fontweight="bold")
        self.ax_mpf2.set_ylabel("Frequency (Hz)", fontsize=22, fontweight="bold")
        self.ax_mpf2.grid(True, linestyle="--", alpha=0.7)
        if show_ma1 or show_ma2 or show_trend:
            self.ax_mpf2.legend(fontsize=9)

        # x-ticks label waktu
        n_labels = labels_comp.size
        if n_labels > 0:
            step = max(1, n_labels // 12)
            tick_idx = np.unique(np.concatenate([np.arange(0, n_labels, step), [n_labels - 1]]))
            for ax in (self.ax_mpf1, self.ax_mpf2):
                ax.set_xticks(tick_idx)
                ax.set_xticklabels(labels_comp[tick_idx], rotation=45, ha="right")

        self.ax_mpf1.tick_params(labelbottom=True)
        self.ax_mpf2.tick_params(labelbottom=True)

        self.canvas_mpf.draw()

    def update_mean_power_plot(self):
        if self.time_labels_export is None:
            return

        L = len(self.time_labels_export)
        if L == 0:
            return

        try:
            ma1 = int(self.ma1_var.get())
        except Exception:
            ma1 = None
        try:
            ma2 = int(self.ma2_var.get())
        except Exception:
            ma2 = None

        label_ma1 = f"MA{ma1}" if ma1 and ma1 > 0 else "MA1"
        label_ma2 = f"MA{ma2}" if ma2 and ma2 > 0 else "MA2"

        show_ma1 = self.show_ma1_var.get()
        show_ma2 = self.show_ma2_var.get()
        show_trend = self.show_trend_var.get()

        # mask drop mean power
        if self.drop_mask_mean is not None and len(self.drop_mask_mean) == L:
            mask = self.drop_mask_mean
        else:
            mask = np.ones(L, dtype=bool)

        kept_idx = np.where(mask)[0]
        if kept_idx.size == 0:
            self.ax_mean1.clear()
            self.ax_mean2.clear()
            self.ax_mean1.set_title("Mean Power (no data after drop)")
            self.ax_mean2.set_title("Mean Power (no data after drop)")
            self.range_adc1_ma1_var.set("Batang Bawah MA1 range: -")
            self.range_adc1_ma2_var.set("Batang Bawah MA2 range: -")
            self.range_adc2_ma1_var.set("Batang Atas MA1 range: -")
            self.range_adc2_ma2_var.set("Batang Atas MA2 range: -")
            self.canvas_mean.draw()
            return

        x = np.arange(kept_idx.size)
        labels_comp = np.array(self.time_labels_export)[kept_idx]

        self.ax_mean1.clear()
        self.ax_mean2.clear()

        # ---------- Batang Atas (ADC2) ----------
        y1_ma1 = None
        y1_ma2 = None

        if show_ma1 and self.mean_power_adc2_ma1 is not None:
            y_top_full = np.asarray(self.mean_power_adc2_ma1, dtype=float)
            y1_ma1 = y_top_full[kept_idx]
            self.ax_mean1.plot(x, y1_ma1, label=f"Mean Power Batang Atas ({label_ma1})", color="orange", linewidth=2, alpha=0.8)

        if show_ma2 and self.mean_power_adc2_ma2 is not None and self.mean_power_adc2_ma2.size > 0:
            M_top = self.mean_power_adc2_ma2.size
            x2_base = np.linspace(0, L - 1, num=M_top)
            y_top_full_ma2 = np.asarray(self.mean_power_adc2_ma2, dtype=float)
            y1_ma2 = np.interp(kept_idx.astype(float), x2_base, y_top_full_ma2)
            self.ax_mean1.plot(x, y1_ma2, label=f"Mean Power Batang Atas ({label_ma2})", color="purple", linewidth=2, alpha=0.8)

        if show_trend and self.mean_power_adc2_ma1 is not None:
            y_tr_top = np.asarray(self.mean_power_adc2_ma1, dtype=float)[kept_idx]
            if y_tr_top.size >= 2:
                coef_top = np.polyfit(x, y_tr_top, 1)
                t_top = np.poly1d(coef_top)(x)
                self.ax_mean1.plot(x, t_top, label="Trend Batang Atas", color="black",
                                   linestyle="--", linewidth=2, alpha=0.8)

        self.ax_mean1.set_title("Mean Power Trend (Batang Atas)", fontsize=22, fontweight="bold")
        self.ax_mean1.set_xlabel("Time", fontsize=22, fontweight="bold")
        self.ax_mean1.set_ylabel("Mean Power (dB)", fontsize=22, fontweight="bold")
        self.ax_mean1.grid(True, linestyle="--", alpha=0.7)
        if show_ma1 or show_ma2 or show_trend:
            self.ax_mean1.legend(fontsize=9)

        # ---------- Batang Bawah (ADC1) ----------
        y2_ma1 = None
        y2_ma2 = None

        if show_ma1 and self.mean_power_adc1_ma1 is not None:
            y_bot_full = np.asarray(self.mean_power_adc1_ma1, dtype=float)
            y2_ma1 = y_bot_full[kept_idx]
            self.ax_mean2.plot(x, y2_ma1, label=f"Mean Power Batang Bawah ({label_ma1})", color="green", linewidth=2, alpha=0.8)

        if show_ma2 and self.mean_power_adc1_ma2 is not None and self.mean_power_adc1_ma2.size > 0:
            M_bot = self.mean_power_adc1_ma2.size
            x2b_base = np.linspace(0, L - 1, num=M_bot)
            y_bot_full_ma2 = np.asarray(self.mean_power_adc1_ma2, dtype=float)
            y2_ma2 = np.interp(kept_idx.astype(float), x2b_base, y_bot_full_ma2)
            self.ax_mean2.plot(x, y2_ma2, label=f"Mean Power Batang Bawah ({label_ma2})", color="purple", linewidth=2, alpha=0.8)

        if show_trend and self.mean_power_adc1_ma1 is not None:
            y_tr2 = np.asarray(self.mean_power_adc1_ma1, dtype=float)[kept_idx]
            if y_tr2.size >= 2:
                coef2 = np.polyfit(x, y_tr2, 1)
                t2 = np.poly1d(coef2)(x)
                self.ax_mean2.plot(x, t2, label="Trend Batang Bawah", color="black",
                                   linestyle="--", linewidth=2, alpha=0.8)

        self.ax_mean2.set_title("Mean Power Trend (Batang Bawah)", fontsize=22, fontweight="bold")
        self.ax_mean2.set_xlabel("Time", fontsize=22, fontweight="bold")
        self.ax_mean2.set_ylabel("Mean Power (dB)", fontsize=22, fontweight="bold")
        self.ax_mean2.grid(True, linestyle="--", alpha=0.7)
        if show_ma1 or show_ma2 or show_trend:
            self.ax_mean2.legend(fontsize=9)

        # x-ticks label waktu
        n_labels = labels_comp.size
        if n_labels > 0:
            step = max(1, n_labels // 12)
            tick_idx = np.unique(np.concatenate([np.arange(0, n_labels, step), [n_labels - 1]]))
            for ax in (self.ax_mean1, self.ax_mean2):
                ax.set_xticks(tick_idx)
                ax.set_xticklabels(labels_comp[tick_idx], rotation=45, ha="right")

        self.ax_mean1.tick_params(labelbottom=True)
        self.ax_mean2.tick_params(labelbottom=True)

        self.canvas_mean.draw()

        # --- Hitung range Mean Power (4 nilai) ---
        def compute_range(arr):
            if arr is None:
                return None
            arr = np.asarray(arr, dtype=float)
            if arr.size == 0 or np.all(np.isnan(arr)):
                return None
            return float(np.nanmax(arr) - np.nanmin(arr))

        r1_ma1 = compute_range(y1_ma1)  # Batang Atas (ADC2)
        r1_ma2 = compute_range(y1_ma2)  # Batang Atas (ADC2)
        r2_ma1 = compute_range(y2_ma1)  # Batang Bawah (ADC1)
        r2_ma2 = compute_range(y2_ma2)  # Batang Bawah (ADC1)

        # r2_* -> Batang Bawah (ADC1), tampil di kolom pertama
        if r2_ma1 is not None:
            self.range_adc1_ma1_var.set(f"Batang Bawah MA1 range: {r2_ma1:.2f}")
        else:
            self.range_adc1_ma1_var.set("Batang Bawah MA1 range: -")

        if r2_ma2 is not None:
            self.range_adc1_ma2_var.set(f"Batang Bawah MA2 range: {r2_ma2:.2f}")
        else:
            self.range_adc1_ma2_var.set("Batang Bawah MA2 range: -")

        # r1_* -> Batang Atas (ADC2), tampil di kolom kedua
        if r1_ma1 is not None:
            self.range_adc2_ma1_var.set(f"Batang Atas MA1 range: {r1_ma1:.2f}")
        else:
            self.range_adc2_ma1_var.set("Batang Atas MA1 range: -")

        if r1_ma2 is not None:
            self.range_adc2_ma2_var.set(f"Batang Atas MA2 range: {r1_ma2:.2f}")
        else:
            self.range_adc2_ma2_var.set("Batang Atas MA2 range: -")


    # ============================================================
    # SAVE DATA & IMAGES
    # ============================================================
    def save_mpf_image(self):
        if self.time_labels_export is None:
            messagebox.showwarning("Belum ada data", "Silakan proses data terlebih dahulu.")
            return
        path = filedialog.asksaveasfilename(
            title="Simpan gambar MPF",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")]
        )
        if not path:
            return
        try:
            self.fig_mpf.savefig(path, dpi=300, bbox_inches="tight")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menyimpan gambar MPF:\n{e}")
            return
        messagebox.showinfo("Berhasil", f"Gambar MPF tersimpan di:\n{path}")

    def save_mean_image(self):
        if self.time_labels_export is None:
            messagebox.showwarning("Belum ada data", "Silakan proses data terlebih dahulu.")
            return
        path = filedialog.asksaveasfilename(
            title="Simpan gambar Mean Power",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")]
        )
        if not path:
            return
        try:
            self.fig_mean.savefig(path, dpi=300, bbox_inches="tight")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menyimpan gambar Mean Power:\n{e}")
            return
        messagebox.showinfo("Berhasil", f"Gambar Mean Power tersimpan di:\n{path}")

    
    def save_data(self):
        if self.time_labels_export is None:
            messagebox.showwarning("Belum ada data", "Silakan proses data terlebih dahulu.")
            return
    
        path = filedialog.asksaveasfilename(
            title="Simpan data ke Excel",
            defaultextension=".xlsx",
            filetypes=[("Excel file", "*.xlsx"), ("All files", "*.*")]
        )
        if not path:
            return
    
        n = len(self.time_labels_export)
    
        # Data yang konsisten dengan plot (MA1)
        mpf1 = np.asarray(self.mpf_adc1_ma1[:n], dtype=float)   # Batang Bawah (ADC1)
        mpf2 = np.asarray(self.mpf_adc2_ma1[:n], dtype=float)   # Batang Atas (ADC2)
        mean1 = np.asarray(self.mean_power_adc2_ma1[:n], dtype=float)  # Mean Power Atas (ADC2)
        mean2 = np.asarray(self.mean_power_adc1_ma1[:n], dtype=float)  # Mean Power Bawah (ADC1)
    
        # ---------- SHEET "Data" ----------
        df_data = pd.DataFrame({
            "Time": self.time_labels_export[:n],
            "MPF_Batang_Atas_Hz": mpf2,
            "MPF_Batang_Bawah_Hz": mpf1,
            "MeanPower_Batang_Atas_dB": mean1,
            "MeanPower_Batang_Bawah_dB": mean2,
        })
    
        # ---------- MASK DROP (supaya Summary mengikuti grafik) ----------
        if getattr(self, "drop_mask_mpf", None) is not None and len(self.drop_mask_mpf) >= n:
            mask_mpf = np.asarray(self.drop_mask_mpf[:n], dtype=bool)
        else:
            mask_mpf = np.ones(n, dtype=bool)
    
        if getattr(self, "drop_mask_mean", None) is not None and len(self.drop_mask_mean) >= n:
            mask_mean = np.asarray(self.drop_mask_mean[:n], dtype=bool)
        else:
            mask_mean = np.ones(n, dtype=bool)
    
        # ---------- SHEET "Summary" (Mean ± SD) ----------
        def _summary_row(series_name: str, arr: np.ndarray, mask: np.ndarray):
            arr = np.asarray(arr, dtype=float)
    
            # All
            mean_all = float(np.nanmean(arr)) if arr.size else np.nan
            sd_all = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else np.nan
            n_all = int(np.sum(np.isfinite(arr)))
    
            # After drop
            kept_vals = arr[mask] if mask is not None and len(mask) == len(arr) else arr
            mean_kept = float(np.nanmean(kept_vals)) if kept_vals.size else np.nan
            sd_kept = float(np.nanstd(kept_vals, ddof=1)) if kept_vals.size > 1 else np.nan
            n_kept = int(np.sum(np.isfinite(kept_vals)))
    
            def fmt(m, s):
                if np.isnan(m) or np.isnan(s):
                    return "-"
                return f"{m:.4f} ± {s:.4f}"
    
            return {
                "Series": series_name,
                "N_All": n_all,
                "Mean_All": mean_all,
                "SD_All": sd_all,
                "Mean±SD_All": fmt(mean_all, sd_all),
                "N_AfterDrop": n_kept,
                "Mean_AfterDrop": mean_kept,
                "SD_AfterDrop": sd_kept,
                "Mean±SD_AfterDrop": fmt(mean_kept, sd_kept),
            }
    
        summary_rows = [
            _summary_row("MPF_Batang_Atas_Hz", mpf2, mask_mpf),
            _summary_row("MPF_Batang_Bawah_Hz", mpf1, mask_mpf),
            _summary_row("MeanPower_Batang_Atas_dB", mean1, mask_mean),
            _summary_row("MeanPower_Batang_Bawah_dB", mean2, mask_mean),
        ]
        df_summary = pd.DataFrame(summary_rows)
    
        # ---------- TULIS KE EXCEL ----------
        try:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                df_data.to_excel(writer, index=False, sheet_name="Data")
                df_summary.to_excel(writer, index=False, sheet_name="Summary")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menyimpan file Excel: {e}")
            return
    
        messagebox.showinfo(
            "Berhasil",
            "Data & ringkasan Mean ± SD tersimpan di:\n"
            f"{path}"
        )
    
if __name__ == "__main__":
    app = MPFMeanPowerGUI()
    app.mainloop()