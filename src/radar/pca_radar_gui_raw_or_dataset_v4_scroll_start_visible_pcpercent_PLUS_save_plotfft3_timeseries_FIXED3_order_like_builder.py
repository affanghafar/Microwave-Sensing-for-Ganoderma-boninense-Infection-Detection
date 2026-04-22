
"""
PCA Radar GUI (2D/3D) - bisa dari:
1) File mentah radar (Excel): hitung Mean Power & MPF -> segmentasi (menit) -> statistik -> PCA
2) Dataset fitur jadi (Excel/CSV): langsung PCA (opsional filter & seleksi fitur)

Catatan:
- Rumus Mean Power & MPF mengikuti `plotfft3_gui_mpf_mean_tabs_final.py`
- Rumus statistik 9 fitur mengikuti `radar_dataset_builder_stats_gui_segmented_v2 (1).py`
- Kolom mentah boleh mengandung COM11/COM12 (auto-detect kolom)

Versi rapi:
- Struktur & komentar diperjelas (step-by-step)
- Logika perhitungan Mean Power/MPF, ekstraksi statistik, dan PCA TIDAK diubah (hasil tetap sama)
"""

from __future__ import annotations

import os
import re
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.ttk as ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PowerTransformer



# ============================================================
# Pre-processing untuk PCA
# ============================================================

TRANSFORM_LABELS = ["Tidak", "Log1p (signed)", "Power (Yeo-Johnson)"]
SCALE_LABELS = ["Tidak", "Z-score", "Robust (median/IQR)"]

def _signed_log1p(x: np.ndarray) -> np.ndarray:
    """Log transform yang aman untuk nilai negatif: sign(x)*log1p(|x|)."""
    return np.sign(x) * np.log1p(np.abs(x))

def _robust_scale(x: np.ndarray) -> np.ndarray:
    """Robust standardization: (x - median) / IQR, IQR = q75-q25."""
    med = np.median(x, axis=0)
    q25 = np.percentile(x, 25, axis=0)
    q75 = np.percentile(x, 75, axis=0)
    iqr = q75 - q25
    iqr[iqr == 0] = 1.0
    return (x - med) / iqr

def preprocess_for_pca(x: np.ndarray, transform_label: str, scale_label: str) -> np.ndarray:
    """Apply transform lalu scaling sesuai pilihan GUI."""
    x2 = x.astype(float, copy=True)

    # transform
    if transform_label == "Log1p (signed)":
        x2 = _signed_log1p(x2)
    elif transform_label == "Power (Yeo-Johnson)":
        # Yeo-Johnson bisa menangani nilai nol/negatif
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        x2 = pt.fit_transform(x2)

    # scaling
    if scale_label == "Z-score":
        x2 = StandardScaler().fit_transform(x2)
    elif scale_label == "Robust (median/IQR)":
        x2 = _robust_scale(x2)

    return x2
# ============================================================
# Konstanta (sesuai dataset_builder)
# ============================================================

CHANNELS = {
    "MPF_Batang_Atas_Hz": "MPF atas (Hz)",
    "MPF_Batang_Bawah_Hz": "MPF bawah (Hz)",
    "MeanPower_Batang_Atas_dB": "Mean Power atas (dB)",
    "MeanPower_Batang_Bawah_dB": "Mean Power bawah (dB)",
}

STAT_NAMES_ORDER = ["mean", "std", "min", "max", "range", "var", "median", "q25", "q75"]

# kolom fitur rapi (urutan)
ORDERED_FEATURES: List[str] = []
for _k, col_label in CHANNELS.items():
    for stat_name in STAT_NAMES_ORDER:
        ORDERED_FEATURES.append(f"{col_label}_{stat_name}")

# warna kelas (sesuai request)
CLASS_COLORS = {
    "sehat": "#2ca02c",
    "ringan": "#ff7f0e",
    "berat": "#d62728",
}


# ============================================================
# Util: parse filename (sesuai dataset_builder)
# ============================================================

@dataclass
class ParsedFilename:
    label: str  # sehat / ringan / berat
    idx: int    # nomor pohon
    fs: int     # frekuensi sampling


def parse_filename(fname: str) -> Optional[ParsedFilename]:
    """
    Mendukung dua format:
    1) Label_Idx_Fs   -> Ringan_1_512.xlsx
    2) Label_Idx      -> Ringan_1.xlsx  (fs default 512)
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_")
    if len(parts) < 2:
        return None

    label_raw = parts[0].strip().lower()
    # normalisasi label
    label_map = {
        "healthy": "sehat",
        "mild": "ringan",
        "severe": "berat",
        "sehat": "sehat",
        "ringan": "ringan",
        "berat": "berat",
    }
    label = label_map.get(label_raw, label_raw)

    # idx
    try:
        idx = int(re.sub(r"\D+", "", parts[1]))
    except Exception:
        return None

    fs = 512
    if len(parts) >= 3:
        try:
            fs = int(re.sub(r"\D+", "", parts[2]))
        except Exception:
            fs = 512

    return ParsedFilename(label=label, idx=idx, fs=fs)


def normalize_label(x: str) -> str:
    s = str(x).strip().lower()
    label_map = {
        "healthy": "sehat",
        "mild": "ringan",
        "severe": "berat",
        "sehat": "sehat",
        "ringan": "ringan",
        "berat": "berat",
    }
    return label_map.get(s, s)


# ============================================================
# Util: deteksi kolom mentah (COM11/COM12 OK)
# ============================================================

def _find_col(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    cols = list(df.columns)
    for pat in patterns:
        rx = re.compile(pat, flags=re.IGNORECASE)
        for c in cols:
            if rx.search(str(c)):
                return c
    return None


def detect_columns_raw(df: pd.DataFrame) -> Dict[str, Optional[np.ndarray]]:
    """
    Kembalikan time, adc1, adc2, v1, v2 sebagai numpy array.
    Mendeteksi variasi nama kolom + COM11/COM12.
    """
    # time
    c_time = _find_col(df, [r"\btime\b.*\(\s*s\s*\)", r"^time\s*\(s\)", r"\btime\b"])
    # ADC counts
    c_adc1 = _find_col(df, [r"\badc1\s*value\b", r"\badc\s*1\b.*value"])
    c_adc2 = _find_col(df, [r"\badc2\s*value\b", r"\badc\s*2\b.*value"])
    # voltage
    c_v1 = _find_col(df, [r"voltage\s*adc1", r"voltage\s*adc\s*1"])
    c_v2 = _find_col(df, [r"voltage\s*adc2", r"voltage\s*adc\s*2"])

    out = {
        "time": df[c_time].to_numpy() if c_time else None,
        "adc1": df[c_adc1].to_numpy() if c_adc1 else None,
        "adc2": df[c_adc2].to_numpy() if c_adc2 else None,
        "v1": df[c_v1].to_numpy() if c_v1 else None,
        "v2": df[c_v2].to_numpy() if c_v2 else None,
        "c_time": c_time,
        "c_adc1": c_adc1,
        "c_adc2": c_adc2,
        "c_v1": c_v1,
        "c_v2": c_v2,
    }
    return out


def detect_sample_size_from_name(path: str, default: int = 512) -> int:
    base = os.path.basename(path)
    m = re.search(r"_([0-9]{3,5})\b", base)
    if m:
        try:
            v = int(m.group(1))
            if v > 0:
                return v
        except Exception:
            pass
    return default


# ============================================================
# MPF + Mean Power (mengikuti plotfft3_gui_mpf_mean_tabs_final.py)
# ============================================================

def compute_mpf_meanpower(time: np.ndarray, sig1: np.ndarray, sig2: np.ndarray, sample_size: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Mengikuti rumus dari plotfft3_gui_mpf_mean_tabs_final.py:
    - seg = seg - mean(seg)  (hilangkan DC)
    - windowing Hanning
    - FFT / sample_size
    - power = abs(FFT)^2 (frekuensi >= 0)
    - power_db = 10*log10(power + 1e-10)
    - MPF = sum(freq * power_linear_norm), dengan power_linear = 10**(power_db/10)
    - Mean Power = mean(power_db)
    - Buang 1 titik terakhir (agar konsisten dengan plot script)
    """
    time = np.asarray(time).astype(float)
    sig1 = np.asarray(sig1).astype(float)
    sig2 = np.asarray(sig2).astype(float)

    # fs dari median delta t
    dt = np.diff(time)
    dt = dt[np.isfinite(dt)]
    if dt.size == 0:
        raise ValueError("Kolom waktu tidak valid (dt kosong).")
    med_dt = float(np.median(dt))
    if med_dt <= 0:
        raise ValueError("Kolom waktu tidak valid (dt <= 0).")
    fs = 1.0 / med_dt

    n = min(len(sig1), len(sig2), len(time))
    n_chunks = n // sample_size
    if n_chunks < 1:
        raise ValueError(f"Data terlalu pendek untuk FFT: n={n}, sample_size={sample_size}")

    # precompute
    window = np.hanning(sample_size)
    freqs = np.fft.fftfreq(sample_size, d=1.0 / fs)
    pos_mask = freqs >= 0
    freqs_pos = freqs[pos_mask]

    mpf1_list: List[float] = []
    mpf2_list: List[float] = []
    mean1_list: List[float] = []
    mean2_list: List[float] = []

    for i in range(n_chunks):
        a = i * sample_size
        b = a + sample_size
        seg1 = sig1[a:b]
        seg2 = sig2[a:b]

        # hilangkan DC
        seg1 = seg1 - np.mean(seg1)
        seg2 = seg2 - np.mean(seg2)

        # windowing
        seg1w = seg1 * window
        seg2w = seg2 * window

        # FFT / sample_size
        fft1 = np.fft.fft(seg1w) / sample_size
        fft2 = np.fft.fft(seg2w) / sample_size

        power1 = np.abs(fft1[pos_mask]) ** 2
        power2 = np.abs(fft2[pos_mask]) ** 2

        power1_db = 10.0 * np.log10(power1 + 1e-10)
        power2_db = 10.0 * np.log10(power2 + 1e-10)

        # MPF
        power1_lin = 10.0 ** (power1_db / 10.0)
        power2_lin = 10.0 ** (power2_db / 10.0)
        s1 = float(np.sum(power1_lin))
        s2 = float(np.sum(power2_lin))
        power1_norm = power1_lin / s1 if s1 != 0 else power1_lin
        power2_norm = power2_lin / s2 if s2 != 0 else power2_lin

        mpf1 = float(np.sum(freqs_pos * power1_norm)) if float(np.sum(power1_norm)) != 0 else 0.0
        mpf2 = float(np.sum(freqs_pos * power2_norm)) if float(np.sum(power2_norm)) != 0 else 0.0

        mpf1_list.append(mpf1)
        mpf2_list.append(mpf2)
        mean1_list.append(float(np.mean(power1_db)) if power1_db.size else 0.0)
        mean2_list.append(float(np.mean(power2_db)) if power2_db.size else 0.0)

    mpf1 = np.asarray(mpf1_list, dtype=float)
    mpf2 = np.asarray(mpf2_list, dtype=float)
    mean1 = np.asarray(mean1_list, dtype=float)
    mean2 = np.asarray(mean2_list, dtype=float)

    # buang 1 titik terakhir
    if len(mpf1) > 1:
        L = len(mpf1) - 1
        mpf1 = mpf1[:L]
        mpf2 = mpf2[:L]
        mean1 = mean1[:L]
        mean2 = mean2[:L]

    return fs, mpf1, mpf2, mean1, mean2


def get_time_labels(n_points: int, start_h: int, start_m: int, chunk_minutes: int = 1) -> List[str]:
    labels = []
    h = int(start_h)
    m = int(start_m)
    for _ in range(n_points):
        labels.append(f"{h:02d}:{m:02d}")
        m += int(chunk_minutes)
        while m >= 60:
            m -= 60
            h += 1
            if h >= 24:
                h = 0
    return labels


# ============================================================
# Statistik 9 fitur (samakan dengan radar_dataset_builder_stats_gui_segmented_v2)
# ============================================================

def compute_stats(arr: np.ndarray) -> Dict[str, float]:
    values = np.asarray(arr, dtype=float)

    # urutan & ddof=0
    out: Dict[str, float] = {}
    try:
        out["mean"] = float(np.mean(values))
    except Exception:
        out["mean"] = float("nan")
    try:
        out["std"] = float(np.std(values))
    except Exception:
        out["std"] = float("nan")
    try:
        out["min"] = float(np.min(values))
    except Exception:
        out["min"] = float("nan")
    try:
        out["max"] = float(np.max(values))
    except Exception:
        out["max"] = float("nan")
    try:
        out["range"] = float(np.max(values) - np.min(values))
    except Exception:
        out["range"] = float("nan")
    try:
        out["var"] = float(np.var(values))
    except Exception:
        out["var"] = float("nan")
    try:
        out["median"] = float(np.median(values))
    except Exception:
        out["median"] = float("nan")
    try:
        out["q25"] = float(np.percentile(values, 25))
    except Exception:
        out["q25"] = float("nan")
    try:
        out["q75"] = float(np.percentile(values, 75))
    except Exception:
        out["q75"] = float("nan")
    return out


# ============================================================
# Build dataset fitur dari file mentah (kolom mengikuti dataset_ref)
# ============================================================

def build_feature_rows_from_raw(
    file_path: str,
    segment_minutes: int,
    step_minutes: Optional[int],
    start_h: int,
    start_m: int,
    use_voltage: bool,
    ma_window: Optional[int],
    include_mpf: bool,
    include_mean: bool,
    include_bawah: bool,
    include_atas: bool,
    stats_selected: List[str],
) -> List[Dict]:
    sample_size = detect_sample_size_from_name(file_path, default=512)

    parsed = parse_filename(os.path.basename(file_path))
    if parsed is None:
        raise ValueError(f"Nama file tidak sesuai format label_idx_fs: {os.path.basename(file_path)}")

    df = pd.read_excel(file_path)
    cols = detect_columns_raw(df)
    time = cols["time"]
    if time is None:
        raise ValueError("Kolom waktu tidak ditemukan (Time (s)).")

    if use_voltage:
        s1 = cols["v1"]
        s2 = cols["v2"]
        if s1 is None or s2 is None:
            raise ValueError("Mode 'Tegangan' dipilih, tetapi kolom Voltage ADC1/ADC2 tidak ditemukan.")
    else:
        s1 = cols["adc1"]
        s2 = cols["adc2"]
        if s1 is None or s2 is None:
            raise ValueError("Mode 'ADC' dipilih, tetapi kolom ADC1 Value / ADC2 Value tidak ditemukan.")

    fs, mpf_adc1, mpf_adc2, mean_adc1, mean_adc2 = compute_mpf_meanpower(time, s1, s2, sample_size)

    time_labels = get_time_labels(len(mpf_adc1), start_h, start_m, chunk_minutes=1)
    # optional Moving Average (MA) sebelum segmentasi/statistik
    if ma_window is not None and int(ma_window) > 1:
        w = int(ma_window)
        base_labels = time_labels
        mpf_adc1 = moving_average(mpf_adc1, w)
        mpf_adc2 = moving_average(mpf_adc2, w)
        mean_adc1 = moving_average(mean_adc1, w)
        mean_adc2 = moving_average(mean_adc2, w)
        # sesuaikan label waktu (centered) agar panjangnya cocok
        if len(base_labels) > w:
            time_labels = base_labels[(w - 1) // 2 : -(w // 2)]
        else:
            time_labels = base_labels


    seg = int(segment_minutes)
    step = int(step_minutes) if step_minutes not in (None, "") else seg
    if seg < 1:
        raise ValueError("Segmentasi (menit) harus >= 1.")
    if step < 1:
        raise ValueError("Step (menit) harus >= 1.")

    series_internal: Dict[str, np.ndarray] = {}

    # MPF
    if include_mpf:
        if include_bawah:
            series_internal["MPF_Batang_Bawah_Hz"] = mpf_adc1  # Mapping default: ADC1 = Batang Bawah
        if include_atas:
            series_internal["MPF_Batang_Atas_Hz"] = mpf_adc2   # Mapping default: ADC2 = Batang Atas
    # Mean Power
    if include_mean:
        if include_bawah:
            series_internal["MeanPower_Batang_Bawah_dB"] = mean_adc1
        if include_atas:
            series_internal["MeanPower_Batang_Atas_dB"] = mean_adc2

    if not series_internal:
        raise ValueError("Tidak ada fitur dipilih. Centang minimal 1: Mean Power/MPF dan Atas/Bawah.")

    rows: List[Dict] = []
    N = len(time_labels)

    seg_idx = 0
    for start in range(0, N - seg + 1, step):
        end = start + seg

        row: Dict = {
            "filename": os.path.basename(file_path),
            "label": parsed.label.capitalize(),  # dataset_ref pakai Kapital awal
            "idx": int(parsed.idx),
            "fs": int(round(fs)),
            "segment_index": int(seg_idx + 1),  # 1-based
            "n_points": int(seg),
            "segment_size": int(seg),
            "time_start": str(time_labels[start]),
            "time_end": str(time_labels[end - 1]),
        }

        # fitur statistik per series
        for internal_key, arr in series_internal.items():
            col_label = CHANNELS[internal_key]
            stats = compute_stats(arr[start:end])
            for stat_name in STAT_NAMES_ORDER:
                if stat_name not in stats_selected:
                    continue
                row[f"{col_label}_{stat_name}"] = stats.get(stat_name, float("nan"))

        rows.append(row)
        seg_idx += 1

    if not rows:
        raise ValueError(
            f"Segmentasi gagal: data hanya punya {N} titik, tetapi segment_size={seg}."
        )

    return rows


# ============================================================
# Load dataset jadi
# ============================================================

def load_dataset_files(paths: List[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext in [".xlsx", ".xls"]:
            df = pd.read_excel(p)
        elif ext in [".csv", ".txt"]:
            df = pd.read_csv(p)
        else:
            raise ValueError(f"Format tidak didukung: {p}")
        dfs.append(df)
    if not dfs:
        raise ValueError("Tidak ada dataset yang dibuka.")
    out = pd.concat(dfs, ignore_index=True)
    return out


def ensure_label_column(df: pd.DataFrame) -> str:
    # cari label col
    candidates = ["label", "kelas", "class", "Class", "Kelas"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: cari kolom yang isinya sehat/ringan/berat
    for c in df.columns:
        vals = set(str(v).strip().lower() for v in df[c].dropna().unique()[:50])
        if {"sehat", "ringan", "berat"}.intersection(vals):
            return c
    raise ValueError("Kolom label/kelas tidak ditemukan.")


def filter_features_from_dataset(df: pd.DataFrame, include_mpf: bool, include_mean: bool, include_bawah: bool, include_atas: bool, stats_selected: List[str]) -> List[str]:
    cols = []
    for internal_key, col_label in CHANNELS.items():
        # filter type & channel
        if internal_key.startswith("MPF") and not include_mpf:
            continue
        if internal_key.startswith("MeanPower") and not include_mean:
            continue
        if "Atas" in internal_key and not include_atas:
            continue
        if "Bawah" in internal_key and not include_bawah:
            continue

        for stat in STAT_NAMES_ORDER:
            if stat not in stats_selected:
                continue
            name = f"{col_label}_{stat}"
            if name in df.columns:
                cols.append(name)
    return cols


# ============================================================
# GUI
# ============================================================

def _set_children_state(widget: tk.Widget, state: str):
    """
    Enable/disable semua child widget (recursive).
    Beberapa widget (Frame/LabelFrame) tidak punya opsi 'state' -> di-skip.
    """
    for child in widget.winfo_children():
        try:
            child.configure(state=state)
        except Exception:
            pass
        # recursive
        _set_children_state(child, state)



class PCARadarGUI(tk.Tk):
    """GUI untuk PCA Radar (2D/3D).

    Dua mode input:
    1) Raw: baca file mentah radar -> hitung Mean Power & MPF -> segmentasi -> statistik -> PCA
    2) Dataset: baca dataset fitur yang sudah jadi -> (opsional filter) -> PCA

    Catatan: Fokus file ini adalah pipeline PCA + visualisasi. Semua perhitungan mengikuti
    script rujukan yang disebut di header docstring.
    """
    def __init__(self):
        super().__init__()
        self.title("PCA Radar (Raw atau Dataset) - Mean Power & MPF (2D/3D)")
        self.geometry("1280x760")

        self.mode_var = tk.StringVar(value="raw")  # raw / dataset

        self.raw_files: List[str] = []
        self.dataset_files: List[str] = []

        self.df_features: Optional[pd.DataFrame] = None
        self.last_feature_cols: List[str] = []
        self.last_label_col: str = "label"

        self.fig = None
        self.canvas = None

        self._build_ui()
        self._update_mode_ui()

    # ---------------- UI ----------------

    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        # kiri: kontrol
        left = ttk.Frame(root)
        left.pack(side="left", fill="y", padx=(0, 10))

        # kanan: plot
        right = ttk.Frame(root)
        right.pack(side="right", fill="both", expand=True)

        # =========================
        # Area atas (dipin): sumber data + input file + tombol start
        # =========================
        top = ttk.Frame(left)
        top.pack(fill="x")

        # Mode
        mode_box = ttk.LabelFrame(top, text="Sumber Data", padding=10)
        mode_box.pack(fill="x", pady=(0, 10))

        ttk.Radiobutton(
            mode_box,
            text="Dari File Mentah (hitung Mean Power/MPF)",
            variable=self.mode_var,
            value="raw",
            command=self._update_mode_ui,
        ).pack(anchor="w")
        ttk.Radiobutton(
            mode_box,
            text="Dari Dataset Jadi (fitur sudah ada)",
            variable=self.mode_var,
            value="dataset",
            command=self._update_mode_ui,
        ).pack(anchor="w")

        # File selection
        file_box = ttk.LabelFrame(top, text="Input", padding=10)
        file_box.pack(fill="x", pady=(0, 10))

        self.btn_pick_raw = ttk.Button(
            file_box, text="Pilih File Mentah (Excel)...", command=self.pick_raw_files
        )
        self.btn_pick_raw.pack(fill="x", pady=(0, 6))

        self.btn_pick_dataset = ttk.Button(
            file_box, text="Pilih Dataset Jadi (Excel/CSV)...", command=self.pick_dataset_files
        )
        self.btn_pick_dataset.pack(fill="x", pady=(0, 6))

        self.lbl_files = ttk.Label(
            file_box, text="Belum ada file dipilih", wraplength=360, justify="left"
        )
        self.lbl_files.pack(fill="x")

        # Actions (selalu terlihat)
        act_box = ttk.LabelFrame(top, text="Aksi", padding=10)
        act_box.pack(fill="x", pady=(0, 10))

        ttk.Button(act_box, text="Start (Proses -> PCA)", command=self.run_pipeline).pack(
            fill="x", pady=(0, 6)
        )
        ttk.Button(act_box, text="Simpan Gambar...", command=self.save_figure).pack(
            fill="x", pady=(0, 6)
        )
        ttk.Button(
            act_box,
            text="Simpan Mean Power & MPF (format PlotFFT3)...",
            command=self.save_mpf_meanpower_plotfft3_format,
        ).pack(fill="x", pady=(0, 6))

        ttk.Button(act_box, text="Simpan Dataset Fitur...", command=self.save_features).pack(
            fill="x"
        )

        # =========================
        # Area bawah (scroll): opsi2 yang panjang
        # =========================
        scroll_host = ttk.Frame(left)
        scroll_host.pack(fill="both", expand=True)

        self._scroll_canvas = tk.Canvas(scroll_host, highlightthickness=0)
        vscroll = ttk.Scrollbar(scroll_host, orient="vertical", command=self._scroll_canvas.yview)
        self._scroll_canvas.configure(yscrollcommand=vscroll.set)

        self._scroll_canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")

        scroll_frame = ttk.Frame(self._scroll_canvas)
        self._scroll_window = self._scroll_canvas.create_window(
            (0, 0), window=scroll_frame, anchor="nw"
        )

        def _on_frame_config(event=None):
            self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all"))

        def _on_canvas_config(event):
            # supaya lebar frame mengikuti lebar canvas
            self._scroll_canvas.itemconfigure(self._scroll_window, width=event.width)

        scroll_frame.bind("<Configure>", _on_frame_config)
        self._scroll_canvas.bind("<Configure>", _on_canvas_config)

        # Mousewheel scroll hanya saat kursor berada di panel kiri
        def _on_mousewheel(event):
            if event.delta:
                self._scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_linux_scroll_up(event):
            self._scroll_canvas.yview_scroll(-3, "units")

        def _on_linux_scroll_down(event):
            self._scroll_canvas.yview_scroll(3, "units")

        def _bind_wheel(_e=None):
            self._scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            self._scroll_canvas.bind_all("<Button-4>", _on_linux_scroll_up)
            self._scroll_canvas.bind_all("<Button-5>", _on_linux_scroll_down)

        def _unbind_wheel(_e=None):
            self._scroll_canvas.unbind_all("<MouseWheel>")
            self._scroll_canvas.unbind_all("<Button-4>")
            self._scroll_canvas.unbind_all("<Button-5>")

        scroll_host.bind("<Enter>", _bind_wheel)
        scroll_host.bind("<Leave>", _unbind_wheel)

        # -------------------------
        # Raw options
        # -------------------------
        raw_box = ttk.LabelFrame(scroll_frame, text="Opsi File Mentah", padding=10)
        raw_box.pack(fill="x", pady=(0, 10))
        self.raw_box = raw_box

        ttk.Label(raw_box, text="Panjang segment (menit):").grid(row=0, column=0, sticky="w")
        self.seg_var = tk.StringVar(value="25")
        ttk.Entry(raw_box, textvariable=self.seg_var, width=8).grid(
            row=0, column=1, sticky="w", padx=(6, 0)
        )

        ttk.Label(raw_box, text="Step (menit, kosong=non-overlap):").grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )
        self.step_var = tk.StringVar(value="")
        ttk.Entry(raw_box, textvariable=self.step_var, width=8).grid(
            row=1, column=1, sticky="w", padx=(6, 0), pady=(6, 0)
        )

        ttk.Label(raw_box, text="Window MA (menit, kosong=tanpa MA):").grid(
            row=2, column=0, sticky="w", pady=(6, 0)
        )
        self.ma_var = tk.StringVar(value="")
        ttk.Entry(raw_box, textvariable=self.ma_var, width=8).grid(
            row=2, column=1, sticky="w", padx=(6, 0), pady=(6, 0)
        )

        ttk.Label(raw_box, text="Waktu mulai (HH:MM):").grid(
            row=3, column=0, sticky="w", pady=(6, 0)
        )
        time_frame = ttk.Frame(raw_box)
        time_frame.grid(row=3, column=1, sticky="w", padx=(6, 0), pady=(6, 0))
        self.start_h_var = tk.StringVar(value="08")
        self.start_m_var = tk.StringVar(value="00")
        ttk.Entry(time_frame, textvariable=self.start_h_var, width=4).pack(side="left")
        ttk.Label(time_frame, text=":").pack(side="left", padx=2)
        ttk.Entry(time_frame, textvariable=self.start_m_var, width=4).pack(side="left")

        self.mode_signal_var = tk.StringVar(value="voltage")
        ttk.Label(raw_box, text="Sumber sinyal:").grid(row=4, column=0, sticky="w", pady=(8, 0))
        sig_frame = ttk.Frame(raw_box)
        sig_frame.grid(row=4, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Radiobutton(sig_frame, text="Tegangan (V)", variable=self.mode_signal_var, value="voltage").pack(anchor="w")
        ttk.Radiobutton(sig_frame, text="ADC (counts)", variable=self.mode_signal_var, value="adc").pack(anchor="w")

        for i in range(2):
            raw_box.grid_columnconfigure(i, weight=0)

        # -------------------------
        # Dataset options
        # -------------------------
        dataset_box = ttk.LabelFrame(scroll_frame, text="Opsi Dataset Jadi", padding=10)
        dataset_box.pack(fill="x", pady=(0, 10))
        self.dataset_box = dataset_box

        self.filter_seg_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            dataset_box,
            text="Filter segment_size",
            variable=self.filter_seg_var,
            command=self._refresh_dataset_filters,
        ).pack(anchor="w")

        seg_line = ttk.Frame(dataset_box)
        seg_line.pack(fill="x", pady=(6, 0))
        ttk.Label(seg_line, text="segment_size:").pack(side="left")
        self.segment_size_combo = ttk.Combobox(seg_line, state="readonly", values=[])
        self.segment_size_combo.pack(side="left", padx=(6, 0), fill="x", expand=True)

        # -------------------------
        # Feature selection (common)
        # -------------------------
        feat_box = ttk.LabelFrame(scroll_frame, text="Fitur untuk PCA", padding=10)
        feat_box.pack(fill="x", pady=(0, 10))

        self.use_mpf_var = tk.BooleanVar(value=True)
        self.use_mean_var = tk.BooleanVar(value=True)
        self.use_atas_var = tk.BooleanVar(value=True)
        self.use_bawah_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(feat_box, text="MPF", variable=self.use_mpf_var).pack(anchor="w")
        ttk.Checkbutton(feat_box, text="Mean Power", variable=self.use_mean_var).pack(anchor="w")
        ttk.Checkbutton(feat_box, text="Batang Atas", variable=self.use_atas_var).pack(anchor="w")
        ttk.Checkbutton(feat_box, text="Batang Bawah", variable=self.use_bawah_var).pack(anchor="w")

        ttk.Separator(feat_box).pack(fill="x", pady=8)

        ttk.Label(feat_box, text="Statistik yang dipakai:").pack(anchor="w")
        self.stat_vars = {}
        stat_grid = ttk.Frame(feat_box)
        stat_grid.pack(fill="x", pady=(4, 0))
        for i, st in enumerate(STAT_NAMES_ORDER):
            var = tk.BooleanVar(value=True)
            self.stat_vars[st] = var
            chk = ttk.Checkbutton(stat_grid, text=st, variable=var)
            chk.grid(row=i // 3, column=i % 3, sticky="w", padx=(0, 12))

        # -------------------------
        # PCA options
        # -------------------------
        pca_box = ttk.LabelFrame(scroll_frame, text="Opsi PCA", padding=10)
        pca_box.pack(fill="x", pady=(0, 10))

        self.dim_var = tk.StringVar(value="2d")
        ttk.Radiobutton(pca_box, text="2D", variable=self.dim_var, value="2d").pack(anchor="w")
        ttk.Radiobutton(pca_box, text="3D", variable=self.dim_var, value="3d").pack(anchor="w")

        # Pre-processing sebelum PCA
        self.transform_var = tk.StringVar(value="Tidak")
        ttk.Label(pca_box, text="Transformasi fitur:").pack(anchor="w", pady=(6, 0))
        ttk.Combobox(
            pca_box, textvariable=self.transform_var, values=TRANSFORM_LABELS, state="readonly", width=24
        ).pack(anchor="w")

        self.scale_var = tk.StringVar(value="Z-score")
        ttk.Label(pca_box, text="Skala fitur:").pack(anchor="w", pady=(6, 0))
        ttk.Combobox(
            pca_box, textvariable=self.scale_var, values=SCALE_LABELS, state="readonly", width=24
        ).pack(anchor="w")

        # Status (di bawah)
        self.status = tk.StringVar(value="Siap.")
        ttk.Label(scroll_frame, textvariable=self.status, wraplength=360, justify="left").pack(fill="x", pady=(10, 0))

        # =========================
        # Plot area
        # =========================
        plot_box = ttk.LabelFrame(right, text="Plot PCA", padding=6)
        plot_box.pack(fill="both", expand=True)

        self.fig = plt.Figure(figsize=(7.2, 6.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_box)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
    def _update_mode_ui(self):
        mode = self.mode_var.get()
        if mode == "raw":
            _set_children_state(self.raw_box, "normal")
            _set_children_state(self.dataset_box, "disabled")
            self.btn_pick_raw.configure(state="normal")
            self.btn_pick_dataset.configure(state="disabled")
        else:
            _set_children_state(self.raw_box, "disabled")
            _set_children_state(self.dataset_box, "normal")
            self.btn_pick_raw.configure(state="disabled")
            self.btn_pick_dataset.configure(state="normal")

        self._refresh_file_label()

    def _refresh_file_label(self):
        mode = self.mode_var.get()
        if mode == "raw":
            if not self.raw_files:
                self.lbl_files.configure(text="Belum ada file mentah dipilih")
            else:
                self.lbl_files.configure(text="File mentah:\n- " + "\n- ".join(os.path.basename(p) for p in self.raw_files[:8]) + (" ..." if len(self.raw_files) > 8 else ""))
        else:
            if not self.dataset_files:
                self.lbl_files.configure(text="Belum ada dataset dipilih")
            else:
                self.lbl_files.configure(text="Dataset:\n- " + "\n- ".join(os.path.basename(p) for p in self.dataset_files[:8]) + (" ..." if len(self.dataset_files) > 8 else ""))

    # ------------- file pickers -------------

    def pick_raw_files(self):
        paths = filedialog.askopenfilenames(
            title="Pilih file mentah radar (.xlsx)",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if not paths:
            return
        self.raw_files = list(paths)
        self.status.set(f"{len(self.raw_files)} file mentah dipilih.")
        self._refresh_file_label()

    def pick_dataset_files(self):
        paths = filedialog.askopenfilenames(
            title="Pilih dataset fitur (.xlsx/.csv)",
            filetypes=[("Excel/CSV", "*.xlsx *.xls *.csv *.txt"), ("All files", "*.*")]
        )
        if not paths:
            return
        self.dataset_files = list(paths)
        self.status.set(f"{len(self.dataset_files)} dataset dipilih.")
        self._refresh_file_label()
        # refresh filter options after load (lazy load later, but we can populate now)
        try:
            df = load_dataset_files(self.dataset_files)
            self._populate_segment_sizes(df)
        except Exception:
            # ignore errors here; will show on run
            pass

    def _populate_segment_sizes(self, df: pd.DataFrame):
        if "segment_size" in df.columns:
            vals = sorted(pd.Series(df["segment_size"]).dropna().unique().tolist())
            self.segment_size_combo["values"] = vals
            if vals:
                self.segment_size_combo.set(vals[0])
        else:
            self.segment_size_combo["values"] = []
            self.segment_size_combo.set("")

    def _refresh_dataset_filters(self):
        # no-op until dataset loaded; handled in run_pipeline
        pass

    # ------------- pipeline -------------

    def _get_stats_selected(self) -> List[str]:
        selected = [k for k, v in self.stat_vars.items() if v.get()]
        if not selected:
            raise ValueError("Pilih minimal 1 statistik.")
        # keep in canonical order
        ordered = [s for s in STAT_NAMES_ORDER if s in selected]
        return ordered

    def run_pipeline(self):
        try:
            # =============================================================
            # PIPELINE UTAMA (GUI) - Step by step
            # -------------------------------------------------------------
            # Step 1) Ambil pilihan user dari GUI:
            #         - jenis fitur (MPF / MeanPower)
            #         - channel (Batang Atas / Batang Bawah)
            #         - statistik yang dipakai
            #         - mode input (raw file vs dataset fitur)
            # Step 2) Bangun / load dataframe fitur:
            #         - mode raw: hitung Mean Power & MPF -> segmentasi -> statistik
            #         - mode dataset: baca file fitur (xlsx/csv)
            # Step 3) Tentukan kolom label + kolom fitur yang valid sesuai pilihan.
            # Step 4) Jalankan PCA (preprocess optional) -> plot 2D/3D.
            # Step 5) Simpan state terakhir (df_features, feature_cols, label_col)
            # =============================================================
            include_mpf = bool(self.use_mpf_var.get())
            include_mean = bool(self.use_mean_var.get())
            include_atas = bool(self.use_atas_var.get())
            include_bawah = bool(self.use_bawah_var.get())
            stats_selected = self._get_stats_selected()

            mode = self.mode_var.get()

            if mode == "raw":
                if not self.raw_files:
                    raise ValueError("Pilih dulu file mentahnya.")
                seg = int(self.seg_var.get())
                step_text = self.step_var.get().strip()
                step = int(step_text) if step_text else None
                ma_text = self.ma_var.get().strip()
                ma_window = int(ma_text) if ma_text else None
                if ma_window is not None and ma_window < 1:
                    raise ValueError('Window MA harus >= 1 atau dikosongkan.')
                start_h = int(self.start_h_var.get())
                start_m = int(self.start_m_var.get())
                use_voltage = (self.mode_signal_var.get() == "voltage")

                rows: List[Dict] = []
                for p in self.raw_files:
                    rr = build_feature_rows_from_raw(
                        p,
                        segment_minutes=seg,
                        step_minutes=step,
                        start_h=start_h,
                        start_m=start_m,
                        use_voltage=use_voltage,
                        ma_window=ma_window,
                        include_mpf=include_mpf,
                        include_mean=include_mean,
                        include_bawah=include_bawah,
                        include_atas=include_atas,
                        stats_selected=stats_selected,
                    )
                    rows.extend(rr)

                df = pd.DataFrame(rows)

                # pastikan kolom fitur yang hilang tetap ada (biar konsisten)
                for col in ORDERED_FEATURES:
                    if col not in df.columns:
                        df[col] = np.nan

                # susun urutan kolom agar IDENTIK dengan radar_dataset_builder_stats_gui_segmented_v2 (meta_cols + ORDERED_FEATURES)
                meta_cols = [
                    "filename",
                    "label",
                    "idx",
                    "fs",
                    "segment_index",
                    "n_points",
                    "segment_size",
                    "time_start",
                    "time_end",
                ]
                for c in meta_cols:
                    if c not in df.columns:
                        df[c] = np.nan
                df = df[meta_cols + ORDERED_FEATURES]

                self.df_features = df

                label_col = "label"
                self.last_label_col = label_col

                # feature cols based on selection
                feature_cols = filter_features_from_dataset(df, include_mpf, include_mean, include_bawah, include_atas, stats_selected)
                if not feature_cols:
                    raise ValueError("Tidak ada kolom fitur yang cocok untuk PCA.")
                self.last_feature_cols = feature_cols

                self.status.set(f"Raw -> fitur: {len(df)} baris, {len(feature_cols)} kolom fitur.")
                self._plot_pca(df, label_col, feature_cols)

            else:
                if not self.dataset_files:
                    raise ValueError("Pilih dulu datasetnya.")

                df = load_dataset_files(self.dataset_files)

                label_col = ensure_label_column(df)
                self.last_label_col = label_col

                # optional filter segment_size
                if bool(self.filter_seg_var.get()) and "segment_size" in df.columns:
                    val = self.segment_size_combo.get()
                    if val != "":
                        df = df[df["segment_size"].astype(str) == str(val)].copy()

                # feature cols based on selection
                feature_cols = filter_features_from_dataset(df, include_mpf, include_mean, include_bawah, include_atas, stats_selected)
                if not feature_cols:
                    raise ValueError("Tidak ada kolom fitur yang cocok di dataset ini. Cek nama kolomnya.")
                self.last_feature_cols = feature_cols

                self.df_features = df

                self.status.set(f"Dataset -> fitur: {len(df)} baris, {len(feature_cols)} kolom fitur.")
                self._plot_pca(df, label_col, feature_cols)

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status.set(f"Error: {e}")

    # ------------- PCA plot -------------

    def _plot_pca(self, df: pd.DataFrame, label_col: str, feature_cols: List[str]):
        # =============================================================
        # PLOT PCA - Step by step
        # -------------------------------------------------------------
        # Step 1) Ambil label (y) lalu normalisasi label ke: sehat/ringan/berat
        # Step 2) Ambil matriks fitur (X) dari kolom terpilih
        # Step 3) Konversi ke numerik, tangani inf/nan, lalu imputasi (mean kolom)
        # Step 4) Preprocess opsional sebelum PCA:
        #         - transform (Tidak / Signed-Log1p / Power Yeo-Johnson)
        #         - scaling (Tidak / Z-score / Robust median-IQR)
        # Step 5) Fit PCA (2D atau 3D) dan hitung explained variance
        # Step 6) Scatter plot per kelas + label PC1/PC2(/PC3) beserta persentase
        # =============================================================
        # prepare X, y
        y_raw = df[label_col].astype(str).tolist()
        y_norm = [normalize_label(v) for v in y_raw]

        X = df[feature_cols].copy()

        # convert numeric and handle inf/nan
        for c in feature_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)
        # imputasi mean kolom
        X = X.fillna(X.mean(numeric_only=True))

        Xv = X.to_numpy(dtype=float)
        # Pre-processing (transform + scaling)
        Xv = preprocess_for_pca(Xv, self.transform_var.get(), self.scale_var.get())

        dim = self.dim_var.get()
        n_comp = 3 if dim == "3d" else 2
        pca = PCA(n_components=n_comp)
        Z = pca.fit_transform(Xv)

        explained = pca.explained_variance_ratio_

        # clear fig
        self.fig.clf()
        if dim == "3d":
            ax = self.fig.add_subplot(111, projection="3d")
        else:
            ax = self.fig.add_subplot(111)

        # title
        title_bits = []
        if self.use_mpf_var.get() and not self.use_mean_var.get():
            title_bits.append("MPF")
        elif self.use_mean_var.get() and not self.use_mpf_var.get():
            title_bits.append("Mean Power")
        else:
            title_bits.append("Mean Power & MPF")
        title = "Plot PCA Radar (" + " + ".join(title_bits) + ")"

        # scatter per class
        classes = ["sehat", "ringan", "berat"]
        for cls in classes:
            mask = [c == cls for c in y_norm]
            if not any(mask):
                continue
            color = CLASS_COLORS.get(cls, "#888888")
            if dim == "3d":
                ax.scatter(Z[mask, 0], Z[mask, 1], Z[mask, 2], s=28, label=cls.capitalize(), c=color)
            else:
                ax.scatter(Z[mask, 0], Z[mask, 1], s=28, label=cls.capitalize(), c=color)

        # labels (dengan persen explained variance)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(f"PC1 ({explained[0]*100:.2f}%)", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"PC2 ({explained[1]*100:.2f}%)", fontsize=12, fontweight="bold")
        if dim == "3d":
            ax.set_zlabel(f"PC3 ({explained[2]*100:.2f}%)", fontsize=12, fontweight="bold")

        ax.legend(fontsize=11)

        self.fig.tight_layout()
        self.canvas.draw()

        # status explained variance
        if dim == "3d":
            self.status.set(
                self.status.get()
                + f"\nExplained variance: PC1={explained[0]*100:.2f}%, PC2={explained[1]*100:.2f}%, PC3={explained[2]*100:.2f}%"
            )
        else:
            self.status.set(
                self.status.get()
                + f"\nExplained variance: PC1={explained[0]*100:.2f}%, PC2={explained[1]*100:.2f}%"
            )

    # ------------- save outputs -------------

    def save_figure(self):
        if self.fig is None:
            return
        path = filedialog.asksaveasfilename(
            title="Simpan gambar PCA",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=300, bbox_inches="tight")
            self.status.set(f"Gambar disimpan: {path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))


    def save_mpf_meanpower_plotfft3_format(self):
        """
        Simpan deret waktu Mean Power & MPF (SEBELUM ekstraksi statistik per segmen)
        dengan format kolom yang sama seperti plotfft3_mpf_savedataall_v2.py:

        Time,
        MPF_Batang_Atas_Hz, MPF_Batang_Bawah_Hz,
        MeanPower_Batang_Atas_dB, MeanPower_Batang_Bawah_dB

        Catatan:
        - Jika Window MA diisi (>1), maka data Mean Power/MPF akan di-MA dahulu (mode 'valid')
          dan Time akan di-trim seperti plotfft3 (center-aligned).
        - File akan disimpan per-raw-file ke folder output yang dipilih.
        """
        try:
            if self.mode_var.get() != "raw":
                raise ValueError("Fungsi ini hanya untuk mode 'File Mentah (raw)'.")
            if not getattr(self, "raw_files", None):
                raise ValueError("Pilih dulu file mentahnya.")

            ma_text = self.ma_var.get().strip()
            ma_window = int(ma_text) if ma_text else None
            if ma_window is not None and ma_window < 1:
                raise ValueError("Window MA harus >= 1 atau dikosongkan.")

            start_h = int(self.start_h_var.get())
            start_m = int(self.start_m_var.get())
            use_voltage = (self.mode_signal_var.get() == "voltage")

            out_dir = filedialog.askdirectory(title="Pilih Folder Output (format PlotFFT3)")
            if not out_dir:
                return

            total = len(self.raw_files)
            ok = 0
            failed = []

            for p in self.raw_files:
                try:
                    df_raw = pd.read_excel(p)
                    cols = detect_columns_raw(df_raw)
                    time = cols.get("time")
                    if time is None:
                        raise ValueError("Kolom waktu tidak ditemukan.")

                    # pilih sinyal sesuai setting GUI (fallback otomatis)
                    if use_voltage:
                        sig1 = cols.get("v1")
                        sig2 = cols.get("v2")
                        if sig1 is None or sig2 is None:
                            sig1 = cols.get("adc1")
                            sig2 = cols.get("adc2")
                    else:
                        sig1 = cols.get("adc1")
                        sig2 = cols.get("adc2")
                        if sig1 is None or sig2 is None:
                            sig1 = cols.get("v1")
                            sig2 = cols.get("v2")

                    if sig1 is None or sig2 is None:
                        raise ValueError("Kolom sinyal (ADC/Voltage) tidak lengkap (butuh CH1 & CH2).")

                    sample_size = detect_sample_size_from_name(p, default=512)

                    fs, mpf_adc1, mpf_adc2, mean_adc1, mean_adc2 = compute_mpf_meanpower(
                        time=time, sig1=sig1, sig2=sig2, sample_size=sample_size
                    )

                    # plotfft3 menampilkan 1 titik per menit (asumsi 1 chunk = 1 menit)
                    time_labels = get_time_labels(len(mpf_adc1), start_h, start_m, chunk_minutes=1)

                    # optional MA (mode='valid' seperti plotfft3)
                    if ma_window is not None and int(ma_window) > 1:
                        w = int(ma_window)
                        mpf_adc1 = moving_average(mpf_adc1, w)
                        mpf_adc2 = moving_average(mpf_adc2, w)
                        mean_adc1 = moving_average(mean_adc1, w)
                        mean_adc2 = moving_average(mean_adc2, w)

                        # trim label agar panjang sama (center-aligned)
                        if len(time_labels) > w:
                            time_labels = time_labels[(w - 1) // 2 : -(w // 2)]

                    L = min(len(time_labels), len(mpf_adc1), len(mpf_adc2), len(mean_adc1), len(mean_adc2))
                    time_labels = time_labels[:L]

                    # Mapping mengikuti plotfft3:
                    # ADC2 = Batang Atas, ADC1 = Batang Bawah
                    df_out = pd.DataFrame({
                        "Time": time_labels,
                        "MPF_Batang_Atas_Hz": mpf_adc2[:L],
                        "MPF_Batang_Bawah_Hz": mpf_adc1[:L],
                        "MeanPower_Batang_Atas_dB": mean_adc2[:L],
                        "MeanPower_Batang_Bawah_dB": mean_adc1[:L],
                    })

                    filename_noext = os.path.splitext(os.path.basename(p))[0]
                    out_path = os.path.join(out_dir, f"{filename_noext}.xlsx")

                    # Hindari overwrite file input kalau user memilih folder yang sama
                    if os.path.abspath(out_path) == os.path.abspath(p):
                        out_path = os.path.join(out_dir, f"{filename_noext}_mpf_meanpower.xlsx")

                    df_out.to_excel(out_path, index=False)
                    ok += 1
                    self.status.set(f"Simpan Mean Power/MPF: {ok}/{total} selesai...")

                except Exception as e:
                    failed.append((p, str(e)))

            if failed:
                msg = "Sebagian file gagal disimpan:\n\n" + "\n".join([f"- {os.path.basename(a)}: {b}" for a,b in failed[:12]])
                if len(failed) > 12:
                    msg += f"\n... dan {len(failed)-12} file lainnya."
                messagebox.showwarning("Selesai (dengan error)", msg)
            else:
                messagebox.showinfo("Selesai", f"Berhasil menyimpan {ok} file (format PlotFFT3).")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_features(self):
        if self.df_features is None:
            messagebox.showinfo("Info", "Belum ada dataset fitur. Jalankan Proses -> PCA dulu.")
            return
        path = filedialog.asksaveasfilename(
            title="Simpan dataset fitur",
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            if path.lower().endswith(".csv"):
                self.df_features.to_csv(path, index=False)
            else:
                self.df_features.to_excel(path, index=False)
            self.status.set(f"Dataset fitur disimpan: {path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))


# ============================================================
# Moving Average (utility)
# ============================================================

def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Simple moving average (mode='valid') seperti plotfft3_gui_mpf_mean_tabs_final.py.
    Jika w <= 1 atau panjang data < w, akan mengembalikan data asli.
    """
    x = np.asarray(x, dtype=float)
    if w is None or w <= 1 or len(x) < w:
        return x.copy()
    return np.convolve(x, np.ones(w, dtype=float) / float(w), mode="valid")


if __name__ == "__main__":
    app = PCARadarGUI()
    app.mainloop()
