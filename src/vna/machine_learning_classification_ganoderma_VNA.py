# === Global font 10pt + centered "Confusion Matrix" title for heatmaps ===
try:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 12,
    })
    try:
        import seaborn as sns  # ensure module is loaded to allow monkey-patching
        if hasattr(sns, "heatmap"):
            _sns_heatmap_orig = sns.heatmap
            def _sns_heatmap_with_title(*args, **kwargs):
                ax = kwargs.get("ax", None)
                out = _sns_heatmap_orig(*args, **kwargs)
                try:
                    _ax = ax if ax is not None else plt.gca()
                    # Centered, bold title slightly above the top of the axes
                    _ax.text(0.5, 1.07, "Confusion Matrix",
                             transform=_ax.transAxes, ha="center", va="bottom",
                             fontsize=10, fontweight="bold")
                except Exception:
                    pass
                return out
            sns.heatmap = _sns_heatmap_with_title
    except Exception:
        pass
except Exception:
    pass
# ========================================================================

# XGboost22_perbagian_FINAL_noCV_9features.py
# =============================================================
#  1 MODEL PER BAGIAN (daun / pelepah / akar) – ES tanpa CV
#  HYPERPARAMETER DI-SET DI KODE (satu set), TANPA GridSearchCV
#  Prediksi 3 bucket + ringkasan akurasi dari nama file
#  UPDATE: 9 FITUR STATISTIK (mean, std, min, max, range, var, median, q25, q75)
# =============================================================

import os
import re
import pickle
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

SUPPRESS_CM_PERPART = False

# --- UI label helpers (display-only English mapping) ---
def _labels_to_english(seq):
    mapping = {'sehat':'Healthy','ringan':'Mild','berat':'Severe'}
    out = []
    for x in seq:
        key = str(x).strip().lower()
        out.append(mapping.get(key, str(x)))
    return out

def _labels_to_indo(seq):
    mapping = {'sehat':'Sehat','ringan':'Ringan','berat':'Berat'}
    out = []
    for x in seq:
        key = str(x).strip().lower()
        out.append(mapping.get(key, str(x)))
    return out

def _part_to_english(x: str) -> str:
    m = {'daun':'Leaf','pelepah':'Frond','akar':'Root'}
    return m.get(str(x).strip().lower(), str(x))

try:
    from xgboost import XGBClassifier
    xgb_ready = True
except Exception:
    xgb_ready = False

# ==========================
#  HYPERPARAMETER DI KODE (1 set)
# ==========================
HP_CONFIG = {
    "daun": {
        "learning_rate":    0.05,
        "max_depth":        4,
        "min_child_weight": 1,
        "subsample":        0.7,
        "colsample_bytree": 0.6,
        "gamma":            0.5,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "n_estimators":     1500,
        "use_gpu":          False,
        "use_cv":           False,
        "cv_splits":        3,
        "param_grid": {
            "learning_rate":    [0.05, 0.06, 0.07],
            "max_depth":        [3, 4, 5],
            "min_child_weight": [1],
            "subsample":        [0.70, 0.85],
            "colsample_bytree": [0.60, 0.70],
            "gamma":            [0.0, 0.5],
            "reg_alpha":        [0.0, 0.1],
            "reg_lambda":       [1.0],
        }
    },
    "akar": {
        "learning_rate":    0.05,
        "max_depth":        4,
        "min_child_weight": 1,
        "subsample":        0.7,
        "colsample_bytree": 0.6,
        "gamma":            0.5,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "use_gpu":          False,
        "use_cv":           False,
        "cv_splits":        3,
        "param_grid": {
            "learning_rate":    [0.05, 0.06, 0.07],
            "max_depth":        [3, 4, 5],
            "min_child_weight": [1],
            "subsample":        [0.70, 0.85],
            "colsample_bytree": [0.60, 0.70],
            "gamma":            [0.0, 0.5],
            "reg_alpha":        [0.0, 0.1],
            "reg_lambda":       [1.0],
        }
    },
    "pelepah": {
        "learning_rate":    0.05,
        "max_depth":        4,
        "min_child_weight": 1,
        "subsample":        0.7,
        "colsample_bytree": 0.6,
        "gamma":            0.5,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "n_estimators":     1500,
        "use_gpu":          False,
        "use_cv":           False,
        "cv_splits":        3,
        "param_grid": {
            "learning_rate":    [0.05, 0.06, 0.07],
            "max_depth":        [3, 4, 5],
            "min_child_weight": [1],
            "subsample":        [0.70, 0.85],
            "colsample_bytree": [0.60, 0.70],
            "gamma":            [0.0, 0.5],
            "reg_alpha":        [0.0, 0.1],
            "reg_lambda":       [1.0],
        }
    },
}

# ==========================
#  UTILITIES / HELPERS
# ==========================

def extract_features_from_file(filepath: str) -> dict:
    """
    Read a measurement .xlsx and compute 9 statistical features per dielectric parameter.
    Features: mean, std, min, max, range, var, median, q25, q75
    """
    df = pd.read_excel(filepath, sheet_name=0)
    kolom_param = ["ε'", "ε\"", "σ (S/m)", "tan(δ)"]
    available = [c for c in kolom_param if c in df.columns]
    if not available:
        raise ValueError(f"Tidak ada kolom parameter yang cocok di: {os.path.basename(filepath)}")
    df = df[available]
    feats = {}
    
    for col in df.columns:
        # Clean data for the column
        col_data = df[col].dropna()
        
        # Calculate 9 statistical features
        feats[f"{col}_mean"]   = col_data.mean()
        feats[f"{col}_std"]    = col_data.std(ddof=1) if len(col_data) > 1 else 0.0
        feats[f"{col}_min"]    = col_data.min()
        feats[f"{col}_max"]    = col_data.max()
        feats[f"{col}_range"]  = col_data.max() - col_data.min()
        feats[f"{col}_var"]    = col_data.var(ddof=1) if len(col_data) > 1 else 0.0
        feats[f"{col}_median"] = col_data.median()
        feats[f"{col}_q25"]    = col_data.quantile(0.25)
        feats[f"{col}_q75"]    = col_data.quantile(0.75)
    
    return feats


def safe_colname(col: str) -> str:
    col = col.replace("'", "_prime").replace('"', "_doubleprime")
    col = re.sub(r"[{}\[\]()]", "_", col)
    col = re.sub(r"[^A-Za-z0-9_\-.]", "_", col)
    return col


def sanitize_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.dropna(axis=1, how='all')
    return X


def simplify_kelas_from_label(label: str) -> str:
    if not isinstance(label, str):
        return str(label)
    parts = label.lower().split('_')
    last = parts[-1] if parts else label.lower()
    if last in ('sehat', 'ringan', 'berat'):
        return last
    if 'berat' in label.lower():
        return 'berat'
    if 'ringan' in label.lower():
        return 'ringan'
    if 'sehat' in label.lower():
        return 'sehat'
    return last

def infer_gt_from_filename(fname: str):
    """Infer ground-truth severity from filename if it contains sehat/ringan/berat."""
    if not isinstance(fname, str):
        return None
    low = fname.lower()
    if 'sehat' in low:  return 'sehat'
    if 'ringan' in low: return 'ringan'
    if 'berat' in low:  return 'berat'
    return None


# ==========================
#  CORE: TRAINING PER-BAGIAN (TANPA CV)
# ==========================

DIELECTRIC_PREFIXES = ["ε'", 'ε"', 'σ (S/m)', 'tan(δ)']  # TANPA Refl.R

def prepare_xy_single_bagian(df: pd.DataFrame, bagian: str):
    if 'bagian' not in df.columns:
        return None, None, "File ekstraksi tidak mengandung kolom 'bagian'."
    if 'label' not in df.columns:
        return None, None, "File ekstraksi tidak mengandung kolom 'label'."

    data_filter = df[df['bagian'].str.lower() == bagian.lower()]
    if data_filter.empty:
        return None, None, f"Tidak ada data untuk bagian '{bagian}'."

    base_cols = [c for c in data_filter.columns if any(c.startswith(p) for p in DIELECTRIC_PREFIXES)]
    if not base_cols:
        return None, None, "Tidak ada kolom fitur dielektrik yang diizinkan."

    X = data_filter[base_cols].copy()
    y = data_filter['label'].astype(str).map(simplify_kelas_from_label)
    X = sanitize_features(X)
    return X, y, None



def run_xgb_training_single_no_cv(data_path: str, bagian: str, hp: dict):
    if not xgb_ready:
        return f"XGBoost belum terinstal. (bagian={bagian})"

    data = pd.read_excel(data_path)
    X, y, err = prepare_xy_single_bagian(data, bagian)
    if err:
        return err

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, stratify=y_enc, test_size=0.20, random_state=42
        )
    except ValueError as e:
        return f"[{bagian}] Error membagi data: {e}"

    # === Inner split for ES (on a subset of TRAIN) ===
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, stratify=y_train, test_size=0.20, random_state=42
    )

    # Imputer median for ES only
    imputer_es = SimpleImputer(strategy="median")
    X_tr_imp  = imputer_es.fit_transform(X_tr)
    X_val_imp = imputer_es.transform(X_val)

    device = "cuda" if bool(hp.get("use_gpu", False)) else "cpu"
    n_estimators_cap = int(hp.get("n_estimators", 1500))

    # === Train with ES to determine best_iteration ===
    model_es = XGBClassifier(
        booster="gbtree",
        objective="multi:softprob",
        n_estimators=n_estimators_cap,
        tree_method="hist",
        eval_metric=["merror","mlogloss"],
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=100,
        device=device,
        learning_rate=float(hp.get("learning_rate", 0.06)),
        max_depth=int(hp.get("max_depth", 5)),
        min_child_weight=int(hp.get("min_child_weight", 1)),
        subsample=float(hp.get("subsample", 0.85)),
        colsample_bytree=float(hp.get("colsample_bytree", 0.7)),
        gamma=float(hp.get("gamma", 0.5)),
        reg_alpha=float(hp.get("reg_alpha", 0.0)),
        reg_lambda=float(hp.get("reg_lambda", 1.0)),
    )
    model_es.fit(X_tr_imp, y_tr, eval_set=[(X_val_imp, y_val)], verbose=False)

    best_iter = getattr(model_es, "best_iteration", None)
    try:
        ev = model_es.evals_result()
    except AttributeError:
        ev = model_es.get_booster().evals_result()

    val_hist = ev.get("validation_0", {})
    arr_merror   = val_hist.get("merror",  [])
    arr_mlogloss = val_hist.get("mlogloss", [])

    best_merror   = None
    best_mlogloss = None
    if best_iter is not None:
        if len(arr_merror)   > best_iter:   best_merror   = arr_merror[best_iter]
        if len(arr_mlogloss) > best_iter:   best_mlogloss = arr_mlogloss[best_iter]
    else:
        # default fallback to keep flow deterministic
        best_iter = min(300, n_estimators_cap)

    # === REFIT on FULL 80% TRAIN (no ES), then test ===
    imputer_full = SimpleImputer(strategy="median")
    X_train_imp  = imputer_full.fit_transform(X_train)
    X_test_imp   = imputer_full.transform(X_test)

    model_refit = XGBClassifier(
        booster="gbtree",
        objective="multi:softprob",
        n_estimators=int(best_iter) + 1,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        device=device,
        learning_rate=float(hp.get("learning_rate", 0.06)),
        max_depth=int(hp.get("max_depth", 5)),
        min_child_weight=int(hp.get("min_child_weight", 1)),
        subsample=float(hp.get("subsample", 0.85)),
        colsample_bytree=float(hp.get("colsample_bytree", 0.7)),
        gamma=float(hp.get("gamma", 0.5)),
        reg_alpha=float(hp.get("reg_alpha", 0.0)),
        reg_lambda=float(hp.get("reg_lambda", 1.0)),
    )
    model_refit.fit(X_train_imp, y_train)

    # === Evaluate on Test with the refit model ===
    y_pred = model_refit.predict(X_test_imp)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    y_test_label = le.inverse_transform(y_test)
    y_pred_label = le.inverse_transform(y_pred)
    report = classification_report(y_test_label, y_pred_label, zero_division=0, digits=4)
    cm = confusion_matrix(y_test_label, y_pred_label, labels=le.classes_)

    # ---- SHAP Global Importance (mean |SHAP|) on Test (Top-10) ----
    shap_pairs = []
    shap_top10_pairs = None
    shap_block_txt = ""
    try:
        import shap
        X_te_np = np.asarray(X_test_imp, dtype=float)
        explainer = shap.TreeExplainer(model_refit)
        shap_values = explainer.shap_values(X_te_np)
        if isinstance(shap_values, list):
            mean_abs = None
            for sv in shap_values:
                sv_np = np.asarray(sv, dtype=float)
                m = np.nan_to_num(np.mean(np.abs(sv_np), axis=0), nan=0.0, posinf=0.0, neginf=0.0)
                mean_abs = m if mean_abs is None else (mean_abs + m)
        else:
            sv_np = np.asarray(shap_values, dtype=float)
            mean_abs = np.nan_to_num(np.mean(np.abs(sv_np), axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        mean_abs = np.asarray(mean_abs, dtype=float).ravel()
        shap_pairs = [(str(col), float(val)) for col, val in zip(list(X.columns), mean_abs.tolist())]
        shap_pairs.sort(key=lambda t: t[1], reverse=True)
        shap_top10_pairs = shap_pairs[:10]
        if shap_top10_pairs:
            lines_txt = []
            for i,(nm,val) in enumerate(shap_top10_pairs):
                try:
                    lines_txt.append(f"  {i+1:2d}. {nm} = {float(val):.6f}")
                except Exception:
                    lines_txt.append(f"  {i+1:2d}. {nm} = {val}")
            shap_block_txt = "Top-10 mean(|SHAP|):\n" + "\n".join(lines_txt) + "\n"
        else:
            shap_block_txt = "Top-10 mean(|SHAP|): not available.\n"
    except Exception as _e:
        shap_block_txt = f"SHAP not available: {_e}\n"
        shap_top10_pairs = None

    # === Plots (SHAP top-10 bar; CM; ES history) ===
    try:
        if (shap_top10_pairs is not None) and (not SUPPRESS_CM_PERPART):
            names = [a for a,_ in shap_top10_pairs]
            vals  = [float(b) for _,b in shap_top10_pairs]
            yy = np.arange(len(names))
            plt.figure(figsize=(8.2, 5.6))
            plt.barh(yy, vals)
            plt.yticks(yy, names, fontsize=10)
            plt.gca().invert_yaxis()
            plt.xlabel("mean(|SHAP|)")
            plt.title(f"SHAP Feature Importance (mean |SHAP|) – {_part_to_english(bagian)}")
            plt.tight_layout()
            plt.show()
    except Exception as _e:
        print("Barplot SHAP failed:", _e)

    if not SUPPRESS_CM_PERPART:
        plt.figure(figsize=(6, 5))
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
                    xticklabels=_labels_to_indo(le.classes_), yticklabels=_labels_to_indo(le.classes_))
        plt.xlabel('Predicted Label'); plt.ylabel('True Label')
        plt.title(f"Confusion Matrix – XGBoost ({(bagian)})")
        plt.tight_layout(); plt.show()

    if arr_merror and arr_mlogloss:
        iters = np.arange(len(arr_mlogloss))
        plt.figure(figsize=(7.2, 4.2))
        plt.plot(iters, arr_merror,   label='merror')
        plt.plot(iters, arr_mlogloss, label='mlogloss')
        if best_iter is not None:
            plt.axvline(best_iter, linestyle='--', alpha=0.6, label=f'best_iter={best_iter}')
        plt.xlabel('Iteration'); plt.ylabel('Metric value')
        plt.title(f'Validation merror vs mlogloss – ({bagian})')
        plt.legend()
        plt.tight_layout(); plt.show()

    hasil_str = (
        f"=== [{bagian.upper()}] XGBoost (1-set HP) + Early Stopping – TANPA CV ===\n"
        f"Statistical Features: 9 features per parameter (mean, std, min, max, range, var, median, q25, q75)\n"
        f"n_estimators upper bound: {n_estimators_cap}\n"
        f"Device: {'GPU (cuda)' if hp.get('use_gpu', False) else 'CPU'}\n"
        f"HP: {{k: hp[k] for k in ['learning_rate','max_depth','min_child_weight','subsample','colsample_bytree','gamma','reg_alpha','reg_lambda']}}\n"
        f"Best Iteration (ES): {best_iter}\n"
        f"Validation merror @best_iter: {best_merror}\n"
        f"Validation mlogloss @best_iter: {best_mlogloss}\n\n"
        f"Test Accuracy: {acc:.4f}  (refit on 80% Train)\n"
        f"Test F1-macro: {f1m:.4f}\n\n"
        "Classification Report (Test):\n"
        f"{report}\n"
        "Confusion Matrix (urutan label sama dengan heatmap):\n"
        f"{cm}\n"
        + shap_block_txt
        + ("="*72) + "\n"
    )

    artifacts = {
        "model": model_refit,
        "label_encoder": le,
        "feature_columns": list(X.columns),
        "imputer": imputer_full,
        "confusion_matrix": cm,
        "eval_history": ev,
        "best_iter": best_iter,
        "label_classes": list(le.classes_),
        "hp": {k: hp[k] for k in ['learning_rate','max_depth','min_child_weight','subsample','colsample_bytree','gamma','reg_alpha','reg_lambda']},
        "n_estimators_upper": n_estimators_cap,
        "shap_mean_abs": dict(shap_pairs) if shap_pairs else None,
        "shap_top10": shap_top10_pairs,
    }

    return hasil_str, artifacts



def run_xgb_training_single_cv_refit80(data_path: str, bagian: str, hp: dict):
    """Per-part XGBoost training with hyperparameter tuning:
    - 80/20 train-test split (test untouched)
    - internal 80/20 split inside train for early stopping (≈64/16/20)
    - GridSearchCV over hp["param_grid"]
    - FINAL refit on full 80% train using best params + best iteration
    """
    if not xgb_ready:
        return f"XGBoost belum terinstal. (bagian={bagian})"

    data = pd.read_excel(data_path)
    X, y, err = prepare_xy_single_bagian(data, bagian)
    if err:
        return err

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, stratify=y_enc, test_size=0.20, random_state=42
        )
    except ValueError as e:
        return f"[{bagian}] Error membagi data: {e}"

    # === Inner split for ES (on a subset of TRAIN) ===
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, stratify=y_train, test_size=0.20, random_state=42
    )

    # Imputer median for ES only
    imputer_es = SimpleImputer(strategy="median")
    X_tr_imp  = imputer_es.fit_transform(X_tr)
    X_val_imp = imputer_es.transform(X_val)

    device = "cuda" if bool(hp.get("use_gpu", False)) else "cpu"
    n_estimators_cap = int(hp.get("n_estimators", 1500))

    # Base estimator (some params will be overridden by GridSearchCV)
    base_xgb = XGBClassifier(
        booster="gbtree",
        objective="multi:softprob",
        n_estimators=n_estimators_cap,
        tree_method="hist",
        eval_metric=["merror", "mlogloss"],
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=100,
        device=device,
    )

    # Param grid
    param_grid = hp.get("param_grid", None)
    if not isinstance(param_grid, dict) or not param_grid:
        param_grid = {
            "learning_rate":    [float(hp.get("learning_rate", 0.06))],
            "max_depth":        [int(hp.get("max_depth", 5))],
            "min_child_weight": [int(hp.get("min_child_weight", 1))],
            "subsample":        [float(hp.get("subsample", 0.85))],
            "colsample_bytree": [float(hp.get("colsample_bytree", 0.7))],
            "gamma":            [float(hp.get("gamma", 0.5))],
            "reg_alpha":        [float(hp.get("reg_alpha", 0.0))],
            "reg_lambda":       [float(hp.get("reg_lambda", 1.0))],
        }

    cv_splits = int(hp.get("cv_splits", 3))
    if cv_splits < 2:
        cv_splits = 2
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    gs = GridSearchCV(
        estimator=base_xgb,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0
    )

    # Grid search on X_tr_imp; early stopping monitored on X_val_imp
    gs.fit(X_tr_imp, y_tr, eval_set=[(X_val_imp, y_val)], verbose=False)

    model_es = gs.best_estimator_
    best_params = gs.best_params_ if hasattr(gs, "best_params_") else {}
    best_cv_score = float(getattr(gs, "best_score_", np.nan))

    # Extract best iteration from early stopping
    best_iter = getattr(model_es, "best_iteration", None)
    try:
        ev = model_es.evals_result()
    except AttributeError:
        ev = model_es.get_booster().evals_result()

    val_hist = ev.get("validation_0", {})
    arr_merror   = val_hist.get("merror",  [])
    arr_mlogloss = val_hist.get("mlogloss", [])

    best_merror   = None
    best_mlogloss = None
    if best_iter is not None:
        if len(arr_merror)   > best_iter:   best_merror   = arr_merror[best_iter]
        if len(arr_mlogloss) > best_iter:   best_mlogloss = arr_mlogloss[best_iter]
    else:
        best_iter = min(300, n_estimators_cap)

    # === FINAL REFIT on FULL 80% TRAIN (no ES), then test ===
    imputer_full = SimpleImputer(strategy="median")
    X_train_imp  = imputer_full.fit_transform(X_train)
    X_test_imp   = imputer_full.transform(X_test)

    # Combine best params + fixed params
    refit_kwargs = dict(best_params) if isinstance(best_params, dict) else {}
    model_refit = XGBClassifier(
        booster="gbtree",
        objective="multi:softprob",
        n_estimators=int(best_iter) + 1,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        device=device,
        **refit_kwargs
    )
    model_refit.fit(X_train_imp, y_train)

    # === Evaluate on Test with the refit model ===
    y_pred = model_refit.predict(X_test_imp)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    y_test_label = le.inverse_transform(y_test)
    y_pred_label = le.inverse_transform(y_pred)
    report = classification_report(y_test_label, y_pred_label, zero_division=0, digits=4)
    cm = confusion_matrix(y_test_label, y_pred_label, labels=le.classes_)

    # ---- SHAP Global Importance (mean |SHAP|) on Test (Top-10) ----
    shap_pairs = []
    shap_top10_pairs = None
    shap_block_txt = ""
    try:
        import shap
        X_te_np = np.asarray(X_test_imp, dtype=float)
        explainer = shap.TreeExplainer(model_refit)
        shap_values = explainer.shap_values(X_te_np)
        if isinstance(shap_values, list):
            mean_abs = None
            for sv in shap_values:
                sv_np = np.asarray(sv, dtype=float)
                m = np.nan_to_num(np.mean(np.abs(sv_np), axis=0), nan=0.0, posinf=0.0, neginf=0.0)
                mean_abs = m if mean_abs is None else (mean_abs + m)
        else:
            sv_np = np.asarray(shap_values, dtype=float)
            mean_abs = np.nan_to_num(np.mean(np.abs(sv_np), axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        mean_abs = np.asarray(mean_abs, dtype=float).ravel()
        shap_pairs = [(str(col), float(val)) for col, val in zip(list(X.columns), mean_abs.tolist())]
        shap_pairs.sort(key=lambda t: t[1], reverse=True)
        shap_top10_pairs = shap_pairs[:10]
        if shap_top10_pairs:
            lines_txt = []
            for i,(nm,val) in enumerate(shap_top10_pairs):
                lines_txt.append(f"  {i+1:2d}. {nm} = {float(val):.6f}")
            shap_block_txt = "Top-10 mean(|SHAP|):\n" + "\n".join(lines_txt) + "\n"
        else:
            shap_block_txt = "Top-10 mean(|SHAP|): not available.\n"
    except Exception as _e:
        shap_block_txt = f"SHAP not available: {_e}\n"
        shap_top10_pairs = None

    # === Plots (SHAP top-10 bar; CM; ES history) ===
    try:
        if (shap_top10_pairs is not None) and (not SUPPRESS_CM_PERPART):
            names = [a for a,_ in shap_top10_pairs]
            vals  = [float(b) for _,b in shap_top10_pairs]
            yy = np.arange(len(names))
            plt.figure(figsize=(8.2, 5.6))
            plt.barh(yy, vals)
            plt.yticks(yy, names, fontsize=10)
            plt.gca().invert_yaxis()
            plt.xlabel("mean(|SHAP|)")
            plt.title(f"SHAP Feature Importance (mean |SHAP|) – {_part_to_english(bagian)} (CV)")
            plt.tight_layout()
            plt.show()
    except Exception as _e:
        print("Barplot SHAP failed:", _e)

    if not SUPPRESS_CM_PERPART:
        plt.figure(figsize=(6, 5))
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
                    xticklabels=_labels_to_indo(le.classes_), yticklabels=_labels_to_indo(le.classes_))
        plt.xlabel('Predicted Label'); plt.ylabel('True Label')
        plt.title(f"Confusion Matrix – XGBoost ({(bagian)})")
        plt.tight_layout(); plt.show()

    if arr_merror and arr_mlogloss:
        iters = np.arange(len(arr_mlogloss))
        plt.figure(figsize=(7.2, 4.2))
        plt.plot(iters, arr_merror,   label='merror')
        plt.plot(iters, arr_mlogloss, label='mlogloss')
        if best_iter is not None:
            plt.axvline(best_iter, linestyle='--', alpha=0.6, label=f'best_iter={best_iter}')
        plt.xlabel('Iteration'); plt.ylabel('Metric value')
        plt.title(f'Validation merror vs mlogloss – ({bagian})')
        plt.legend()
        plt.tight_layout(); plt.show()

    hasil_str = (
        f"=== [{bagian.upper()}] XGBoost – CV GridSearch + Early Stopping + Refit80 ===\n"
        f"Statistical Features: 9 features per parameter (mean, std, min, max, range, var, median, q25, q75)\n"
        f"n_estimators upper bound: {n_estimators_cap}\n"
        f"Device: {'GPU (cuda)' if hp.get('use_gpu', False) else 'CPU'}\n"
        f"CV folds: {cv_splits} | CV best score (accuracy): {best_cv_score:.4f}\n"
        f"Best Params: {best_params}\n"
        f"Best Iteration (ES): {best_iter}\n"
        f"Validation merror @best_iter: {best_merror}\n"
        f"Validation mlogloss @best_iter: {best_mlogloss}\n\n"
        f"Test Accuracy: {acc:.4f}  (refit on 80% Train)\n"
        f"Test F1-macro: {f1m:.4f}\n\n"
        "Classification Report (Test):\n"
        f"{report}\n"
        "Confusion Matrix (urutan label sama dengan heatmap):\n"
        f"{cm}\n"
        + shap_block_txt
        + ("="*72) + "\n"
    )

    artifacts = {
        "model": model_refit,
        "label_encoder": le,
        "feature_columns": list(X.columns),
        "imputer": imputer_full,
        "confusion_matrix": cm,
        "eval_history": ev,
        "best_iter": best_iter,
        "label_classes": list(le.classes_),
        "hp": {
            "best_params": best_params,
            "cv_splits": cv_splits,
            "best_cv_score": best_cv_score,
        },
        "n_estimators_upper": n_estimators_cap,
        "shap_mean_abs": dict(shap_pairs) if shap_pairs else None,
        "shap_top10": shap_top10_pairs,
    }

    return hasil_str, artifacts


# ==========================
#  GUI
# ==========================

class MLXGBGUI:
    def __init__(self, master):
        self.master = master
        master.title("VNA Oil-Palm Analyzer – Per‑Part XGBoost (ES, no CV) - 9 Features")
        master.geometry("1180x780")
        master.minsize(1040, 700)

        self.models = {'daun': None, 'akar': None, 'pelepah': None}
        self.last_summary_text = ""

        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f7f7f7')
        self.style.configure('TLabel', background='#f7f7f7', font=('Segoe UI', 9))
        self.style.configure('TButton', font=('Segoe UI', 9), padding=4)
        self.style.configure('TNotebook.Tab', font=('Segoe UI', 9, 'bold'), padding=[8, 4])

        container = ttk.Frame(master)
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        nb = ttk.Notebook(container)
        nb.pack(fill=tk.BOTH, expand=True)
        self.tab_ml = ttk.Frame(nb)
        nb.add(self.tab_ml, text='Training & Prediction (Per‑Part)')

        self.build_ml_tab()

    def build_ml_tab(self):
        # 1) File input training
        s1 = ttk.LabelFrame(self.tab_ml, text="1. Training Input Files (combined feature‑extraction .xlsx)", padding=(8,4))
        s1.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(s1, text="Feature‑Extraction File (.xlsx):").grid(row=0, column=0, padx=4, sticky='e')
        self.mlfile_var = tk.StringVar()
        ttk.Entry(s1, textvariable=self.mlfile_var, width=78).grid(row=0, column=1, padx=4)
        ttk.Button(s1, text="Browse", command=self.browse_mlfile).grid(row=0, column=2, padx=4)

        # 2) Pilih bagian
        s2 = ttk.LabelFrame(self.tab_ml, text="2. Select Plant Parts – one model per box (results shown per box)", padding=(8,4))
        s2.pack(fill=tk.X, padx=4, pady=4)
        self.var_daun = tk.BooleanVar(value=True)
        self.var_akar = tk.BooleanVar(value=False)
        self.var_pelepah = tk.BooleanVar(value=False)
        ttk.Checkbutton(s2, text="Leaf", variable=self.var_daun).grid(row=0, column=0, padx=6, sticky='w')
        ttk.Checkbutton(s2, text="Root", variable=self.var_akar).grid(row=0, column=1, padx=6, sticky='w')
        ttk.Checkbutton(s2, text="Frond", variable=self.var_pelepah).grid(row=0, column=2, padx=6, sticky='w')

        # 3) Training, Evaluasi, Save/Load
        s4 = ttk.LabelFrame(self.tab_ml, text="3. Training, Evaluation, and Model", padding=(8,4))
        s4.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        topbar = ttk.Frame(s4); topbar.pack(fill=tk.X, pady=(4,6))
        ttk.Button(topbar, text="Per‑Part Train", command=self.run_ml_perbagian).pack(side=tk.LEFT, padx=4)

        # CV tuning controls (optional)
        self.use_cv_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(topbar, text="Enable CV tuning (GridSearch)", variable=self.use_cv_var).pack(side=tk.LEFT, padx=10)
        ttk.Label(topbar, text="CV folds:").pack(side=tk.LEFT)
        self.cv_splits_var = tk.IntVar(value=3)
        try:
            ttk.Spinbox(topbar, from_=2, to=10, width=4, textvariable=self.cv_splits_var).pack(side=tk.LEFT, padx=4)
        except Exception:
            tk.Spinbox(topbar, from_=2, to=10, width=4, textvariable=self.cv_splits_var).pack(side=tk.LEFT, padx=4)
        ttk.Button(topbar, text="Save Bundle (.pkl)…", command=self.save_bundle).pack(side=tk.LEFT, padx=4)
        ttk.Button(topbar, text="Load Bundle (.pkl)…", command=self.load_bundle).pack(side=tk.LEFT, padx=4)

        self.result_txt = scrolledtext.ScrolledText(s4, width=120, height=18, font=('Consolas', 10), wrap='word')
        self.result_txt.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # 4) Prediksi: 3 Bucket (Leaf / Frond / Root)
        s5 = ttk.LabelFrame(self.tab_ml, text="4. Predict Files (.xlsx, Sheet1) – select files per part", padding=(8,4))
        s5.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        frm_daun = ttk.LabelFrame(s5, text="Leaf");    frm_daun.grid(row=0, column=0, sticky='nsew', padx=4, pady=4)
        frm_pep  = ttk.LabelFrame(s5, text="Frond");   frm_pep.grid(row=0, column=1, sticky='nsew', padx=4, pady=4)
        frm_akr  = ttk.LabelFrame(s5, text="Root");    frm_akr.grid(row=0, column=2, sticky='nsew', padx=4, pady=4)
        for i in range(3):
            s5.grid_columnconfigure(i, weight=1)
        s5.grid_rowconfigure(0, weight=1)

        def build_bucket(frame, attr_prefix, bagian_name):
            lb = tk.Listbox(frame, selectmode=tk.EXTENDED, height=7)
            lb.grid(row=0, column=0, columnspan=2, sticky='nsew', padx=4, pady=4)
            sb = ttk.Scrollbar(frame, orient='vertical', command=lb.yview)
            sb.grid(row=0, column=2, sticky='ns')
            lb.config(yscrollcommand=sb.set)

            btns = ttk.Frame(frame); btns.grid(row=1, column=0, columnspan=3, sticky='w', padx=4, pady=2)

            def add_files():
                paths = filedialog.askopenfilenames(title=f"Select {bagian_name} files (.xlsx) – multi‑select allowed",
                                                    filetypes=[("Excel files", "*.xlsx")])
                if not paths:
                    return
                if isinstance(paths, str):
                    paths = self.master.tk.splitlist(paths)
                existing = set(lb.get(0, tk.END))
                for p in paths:
                    if p not in existing:
                        lb.insert(tk.END, p)
                        existing.add(p)

            def remove_sel():
                sel = list(lb.curselection())
                if not sel:
                    messagebox.showinfo("Info", f"Select {bagian_name} items to remove (Ctrl/Shift for multi)." )
                    return
                for idx in reversed(sel):
                    lb.delete(idx)

            def clear_all():
                lb.delete(0, tk.END)

            ttk.Button(btns, text="Add Files…", command=add_files).grid(row=0, column=0, padx=4)
            ttk.Button(btns, text="Remove Selected", command=remove_sel).grid(row=0, column=1, padx=4)
            ttk.Button(btns, text="Clear", command=clear_all).grid(row=0, column=2, padx=4)

            frame.grid_rowconfigure(0, weight=1)
            frame.grid_columnconfigure(0, weight=1)

            setattr(self, f"{attr_prefix}_listbox", lb)

        build_bucket(frm_daun, "daun", "Leaf")
        build_bucket(frm_pep,  "pelepah", "Frond")
        build_bucket(frm_akr,  "akar", "Root")

        bottom = ttk.Frame(s5); bottom.grid(row=1, column=0, columnspan=3, sticky='w', padx=4, pady=6)
        ttk.Button(bottom, text="Predict All Buckets", command=self.run_predict_files).grid(row=0, column=0, padx=4)

        self.btn_save_pred = ttk.Button(bottom, text="Save Results…", command=self.save_pred_results, state='disabled')
        self.btn_save_pred.grid(row=0, column=1, padx=8)

        self.pred_result_box = scrolledtext.ScrolledText(s5, width=120, height=12, font=('Consolas', 10), wrap='word')
        self.pred_result_box.grid(row=2, column=0, columnspan=3, sticky='nsew', padx=4, pady=4)
        self.pred_result_box.configure(state='disabled')

        s5.grid_rowconfigure(2, weight=1)

    # ---------- Helpers ----------
    def browse_mlfile(self):
        path = filedialog.askopenfilename(title="Select feature‑extraction file (.xlsx)",
                                          filetypes=[("Excel files", "*.xlsx")])
        if path:
            self.mlfile_var.set(path)

    def _gather_bucket_files(self):
        # Return list of tuples: (filepath, bagian)
        items = []
        for bagian, attr in (('daun','daun_listbox'), ('pelepah','pelepah_listbox'), ('akar','akar_listbox')):
            lb = getattr(self, attr, None)
            if lb is None: 
                continue
            files = list(lb.get(0, tk.END))
            for fp in files:
                items.append((fp, bagian))
        return items

    # ---------- Training Per-Bagian ----------
    def run_ml_perbagian(self):
        data_path = self.mlfile_var.get()
        pilih = []
        if self.var_daun.get(): pilih.append('daun')
        if self.var_akar.get(): pilih.append('akar')
        if self.var_pelepah.get(): pilih.append('pelepah')

        if not xgb_ready:
            messagebox.showerror("Error", "XGBoost is not installed. Please install the 'xgboost' package first.")
            return
        if not data_path or not os.path.exists(data_path):
            messagebox.showerror("Error", "No feature‑extraction file selected!")
            return
        if not pilih:
            messagebox.showerror("Error", "Check at least one plant part!")
            return
        
        # --- Combined CM mode: if all three parts are selected, suppress per-part CM windows
        global SUPPRESS_CM_PERPART
        show_combined_cm = (set(pilih) == {'daun','pelepah','akar'})
        SUPPRESS_CM_PERPART = show_combined_cm

        use_cv_gui = bool(getattr(self, "use_cv_var", tk.BooleanVar(value=False)).get())
        try:
            cv_splits_gui = int(getattr(self, "cv_splits_var", tk.IntVar(value=3)).get())
        except Exception:
            cv_splits_gui = 3
        cv_splits_gui = max(2, cv_splits_gui)

        # Reset and announce
        self.result_txt.delete("1.0", tk.END)
        mode_txt = "CV GridSearch + ES + Refit80" if use_cv_gui else "1-set HP + ES (no CV)"
        self.result_txt.insert(tk.END, f"Starting Training ({mode_txt}) with 9 Statistical Features...\n\n")
        teks_all, trained_any = [], False

        for bagian in pilih:
            hp = dict(HP_CONFIG.get(bagian, HP_CONFIG["daun"]))
            hp["use_cv"] = use_cv_gui
            hp["cv_splits"] = cv_splits_gui
            try:
                if use_cv_gui:
                    res = run_xgb_training_single_cv_refit80(
                        data_path=data_path,
                        bagian=bagian,
                        hp=hp
                    )
                else:
                    res = run_xgb_training_single_no_cv(
                        data_path=data_path,
                        bagian=bagian,
                        hp=hp
                    )
            except Exception as e:
                self.result_txt.insert(tk.END, f"[{bagian}] FAILED: {e}\n\n")
                continue

            if isinstance(res, str):
                self.result_txt.insert(tk.END, f"[{bagian}] {res}\n\n")
                continue

            summary_text, artifacts = res
            self.models[bagian] = artifacts
            teks_all.append(summary_text)
            trained_any = True

        if trained_any:
            # If all three parts are selected, show all CMs in one figure with (a), (b), (c)
            if set(pilih) == {'daun','pelepah','akar'}:
                try:
                    import seaborn as sns
                    import numpy as np
                    SUPPRESS_CM_PERPART = True  # avoid per-part windows

                    parts_order = ['daun','pelepah','akar']
                    titles_en = {'daun':'Leaf','pelepah':'Frond','akar':'Root'}
                    letters = ['a','b','c']
                    mats = []
                    classes_list = []

                    for b in parts_order:
                        art = self.models.get(b)
                        if not art: break
                        cm = art.get('confusion_matrix', None)
                        cls = art.get('label_classes', None)
                        if cm is None or cls is None: break
                        mats.append(cm)
                        classes_list.append(cls)

                    if len(mats) == 3:
                        fig = plt.figure(figsize=(10.6, 7.4))
                        # Fixed axis boxes (fraction of figure)
                        AX_W, AX_H = 0.35, 0.35
                        X_LEFT = 0.08
                        X_RIGHT = 1.0 - X_LEFT - AX_W    # symmetric
                        Y_TOP  = 0.56
                        X_MID  = 0.5 - AX_W/2.0
                        Y_BOT  = 0.12

                        ax_tl = fig.add_axes([X_LEFT,  Y_TOP, AX_W, AX_H])
                        ax_tr = fig.add_axes([X_RIGHT, Y_TOP, AX_W, AX_H])
                        ax_bc = fig.add_axes([X_MID,   Y_BOT, AX_W, AX_H])
                        axes  = [ax_tl, ax_tr, ax_bc]

                        # Plot each CM
                        for ax, cm, cls in zip(axes, mats, classes_list):
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
                                        xticklabels=_labels_to_indo(cls),
                                        yticklabels=_labels_to_indo(cls),
                                        ax=ax)
                            ax.set_xlabel('Predicted Label', labelpad=0)
                            ax.set_ylabel('True Label')
                            ax.set_title("")

                        # Figure-level captions
                        titles_en = {'daun':'Leaf','pelepah':'Frond','akar':'Root'}
                        letters   = ['a','b','c']
                        for ax, letter, b in zip(axes, letters, parts_order):
                            pos = ax.get_position()
                            fig.text(pos.x0 + pos.width/2.0, pos.y0 - 0.060, f"({letter}) {titles_en.get(b, b)}",
                                     ha='center', va='top', fontweight='bold', fontsize=10)

                        plt.show()

                except Exception as e:
                    print("Combined CM plot failed:", e)
                finally:
                    SUPPRESS_CM_PERPART = False

            # === SHAP (mean |SHAP|) – layout like CM (2 top, 1 bottom) ===
            try:
                import numpy as np
                parts_order = ['daun','pelepah','akar']
                titles_en = {'daun':'Leaf','pelepah':'Frond','akar':'Root'}
                shap_lists = []
                vmax = 0.0
                for b in parts_order:
                    art = self.models.get(b, {})
                    top10 = art.get('shap_top10') if isinstance(art, dict) else None
                    if not top10:
                        shap_lists.append(None)
                        continue
                    shap_lists.append(top10)
                    mx = max((v for _,v in top10), default=0.0)
                    if mx > vmax: vmax = mx
                if any(shap_lists):
                    fig = plt.figure(figsize=(10.6, 7.4))
                    AX_W, AX_H = 0.35, 0.35
                    X_LEFT = 0.08
                    X_RIGHT = 1.0 - X_LEFT - AX_W
                    Y_TOP  = 0.56
                    X_MID  = 0.5 - AX_W/2.0
                    Y_BOT  = 0.12
                    ax_tl = fig.add_axes([X_LEFT,  Y_TOP, AX_W, AX_H])
                    ax_tr = fig.add_axes([X_RIGHT, Y_TOP, AX_W, AX_H])
                    ax_bc = fig.add_axes([X_MID,   Y_BOT, AX_W, AX_H])
                    axes  = [ax_tl, ax_tr, ax_bc]
                    for ax, top10, b in zip(axes, shap_lists, parts_order):
                        ax.set_title("")
                        if not top10:
                            ax.text(0.5, 0.5, "No SHAP info", ha="center", va="center")
                            ax.set_xticks([]); ax.set_yticks([])
                            continue
                        names = [n for n,_ in top10]
                        vals  = [v for _,v in top10]
                        yy = np.arange(len(names))
                        ax.barh(yy, vals)
                        ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=10)
                        ax.invert_yaxis()
                        ax.set_xlim(0, vmax * 1.05 if vmax>0 else 1.0)
                        ax.set_xlabel("mean(|SHAP|)")
                    letters = ['a','b','c']
                    for ax, letter, b in zip(axes, letters, parts_order):
                        pos = ax.get_position()
                        fig.text(pos.x0 + pos.width/2.0, pos.y0 - 0.060, f"({letter}) {titles_en.get(b, b)}",
                                 ha='center', va='top', fontweight='bold', fontsize=10)
                    fig.suptitle("SHAP Feature Importance (mean |SHAP|)", y=0.99, fontsize=10, fontweight='bold')
                    plt.show()
            except Exception as e:
                print("Combined SHAP plot failed:", e)

            gabung = "\n".join(teks_all)
            self.last_summary_text = gabung
            self.result_txt.delete("1.0", tk.END)
            self.result_txt.insert(tk.END, gabung)
        else:
            self.result_txt.insert(tk.END, "No model was successfully trained.\n")

    # ---------- Save/Load Bundle (.pkl) ----------
    def save_bundle(self):
        if not any(self.models.values()):
            messagebox.showwarning("No Model Loaded", "Train at least one model first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Per‑Part Model Bundle",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            initialfile="xgb_perbagian_9features_bundle.pkl"
        )
        if not path:
            return
        try:
            with open(path, "wb") as f:
                pickle.dump({"models": self.models, "summary_text": self.last_summary_text}, f)
            messagebox.showinfo("Success", f"Bundle saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Gagal Menyimpan", f"Terjadi kesalahan saat menyimpan bundle:\n{e}")

    def load_bundle(self):
        path = filedialog.askopenfilename(
            title="Load Bundle Model Per-Bagian",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "rb") as f:
                bundle = pickle.load(f)

            md = bundle.get("models", {})
            if not isinstance(md, dict) or not any(md.values()):
                raise ValueError("The bundle does not contain valid models.")

            self.models = {'daun': None, 'akar': None, 'pelepah': None}
            for k in ['daun', 'akar', 'pelepah']:
                if k in md and md[k] is not None:
                    self.models[k] = md[k]

            self.last_summary_text = bundle.get("summary_text", "")

            self.result_txt.delete("1.0", tk.END)
            if self.last_summary_text:
                self.result_txt.insert(tk.END, self.last_summary_text)
            else:
                terisi = [k for k,v in self.models.items() if v is not None]
                self.result_txt.insert(tk.END, "Bundle loaded. Available models: " + ", ".join(terisi))

            # Tampilkan plot untuk setiap model
            for bagian, art in self.models.items():
                if not art: 
                    continue
                cm = art.get("confusion_matrix", None)
                classes = art.get("label_classes", None)
                if isinstance(cm, np.ndarray) and classes is not None:
                    import seaborn as sns
                    plt.figure(figsize=(6,5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
                                xticklabels=_labels_to_indo(classes), yticklabels=_labels_to_indo(classes))
                    plt.xlabel('Predicted Label'); plt.ylabel('True Label')
                    plt.title(f"Confusion Matrix – (Loaded) [{_part_to_english(bagian)}]")
                    plt.tight_layout(); plt.show()

                ev = art.get("eval_history", None)
                mp, ms, bi = "merror", "mlogloss", art.get("best_iter", None)
                if isinstance(ev, dict) and "validation_0" in ev:
                    val_hist = ev.get("validation_0", {})
                    arr_primary   = val_hist.get(mp,  [])
                    arr_secondary = val_hist.get(ms, [])
                    if arr_primary and arr_secondary:
                        iters = np.arange(len(arr_secondary))
                        plt.figure(figsize=(7.2, 4.2))
                        plt.plot(iters, arr_primary,   label=mp)
                        plt.plot(iters, arr_secondary, label=ms)
                        if bi is not None:
                            plt.axvline(bi, linestyle='--', alpha=0.6, label=f'best_iter={bi}')
                        plt.xlabel('Iteration'); plt.ylabel('Metric value')
                        plt.title(f'Validation {mp} vs {ms} – (Loaded) [{bagian}]')
                        plt.legend(); plt.tight_layout(); plt.show()

            messagebox.showinfo("Success", "Bundle loaded successfully and the summary was displayed.")
        except Exception as e:
            messagebox.showerror("Gagal Memuat", f"Terjadi kesalahan saat memuat bundle:\n{e}")

    # ---------- Prediksi (3 bucket) + Akurasi ----------
    def run_predict_files(self):
        if not any(self.models.values()):
            messagebox.showwarning("No Model Loaded", "Latih atau Load bundle model terlebih dahulu sebelum prediksi.")
            return

        items = self._gather_bucket_files()
        if not items:
            messagebox.showwarning("Tidak Ada File", "Daftar file di semua bucket masih kosong. Tambahkan file terlebih dahulu.")
            return

        self.last_pred_df = None
        self.last_pred_dir = None
        self.btn_save_pred.config(state='disabled')

        results, rows = [], []
        total, ok, fail = len(items), 0, 0

        for (fp, bagian) in items:
            try:
                if not os.path.exists(fp):
                    raise FileNotFoundError("File tidak ditemukan")

                art = self.models.get(bagian)
                if not art:
                    raise ValueError(f"Model untuk bagian '{bagian}' belum tersedia. Latih atau Load bundle yang berisi model tersebut.")

                fitur = extract_features_from_file(fp)
                row = fitur.copy()
                row['filename'] = os.path.basename(fp)

                df_pred = pd.DataFrame([row])
                keep_cols = [c for c in df_pred.columns if any(c.startswith(p) for p in DIELECTRIC_PREFIXES)]
                Xp = df_pred[keep_cols].copy()
                Xp = sanitize_features(Xp)

                feat_cols = art['feature_columns']
                orig_cols = list(Xp.columns)
                overlap = sorted(set(feat_cols) & set(orig_cols))
                if not overlap:
                    raise ValueError("Fitur tidak cocok dengan model—tidak ada kolom yang overlap dengan fitur training. Pastikan format ekstraksi sama.")

                Xp = Xp.reindex(columns=feat_cols)  # urutan sesuai training
                Xp = Xp.astype('float64')
                Xp_imp = art['imputer'].transform(Xp)

                model = art['model']
                le = art['label_encoder']

                y_pred_enc = model.predict(Xp_imp)
                y_pred = le.inverse_transform(y_pred_enc)[0]
                kelas_sederhana = simplify_kelas_from_label(y_pred)

                prob_val = None
                try:
                    proba = model.predict_proba(Xp_imp)[0]
                    classes = list(le.classes_)
                    idx = classes.index(y_pred)
                    prob_val = float(proba[idx])
                except Exception:
                    pass

                results.append(f"{os.path.basename(fp)}  →  [{bagian}]  {kelas_sederhana}" + (f" ({prob_val:.3f})" if prob_val is not None else ""))

                gt = infer_gt_from_filename(os.path.basename(fp))
                rows.append({
                    'filename': os.path.basename(fp),
                    'bagian_infer': bagian,
                    'label_prediksi_asli': str(y_pred),
                    'kelas_pred': str(kelas_sederhana),
                    'probabilitas': prob_val,
                    'gt_from_filename': gt
                })
                ok += 1

            except Exception as e:
                results.append(f"{os.path.basename(fp)}  →  FAILED: {e}")
                fail += 1

        ringkas = [
            f"Predict Files (total {total} files from 3 buckets)",
            f"Success: {ok} | Gagal: {fail}",
            "-"*60
        ] + results + ["-"*60]

        # === Hitung akurasi dari GT yang diinfer dari nama file ===
        try:
            df_eval = pd.DataFrame(rows) if rows else pd.DataFrame()
            has_gt = df_eval['gt_from_filename'].notna().sum() if not df_eval.empty and 'gt_from_filename' in df_eval.columns else 0
            if has_gt > 0:
                df_sub = df_eval.dropna(subset=['gt_from_filename']).copy()
                df_sub['benar'] = (df_sub['kelas_pred'].astype(str) == df_sub['gt_from_filename'].astype(str)).astype(int)
                benar = int(df_sub['benar'].sum())
                salah = int(len(df_sub) - benar)
                akurasi = 100.0 * benar / max(1, len(df_sub))
                ringkas.append(f"Evaluasi dari nama file → N={len(df_sub)} | Benar={benar} | Salah={salah} | Akurasi={akurasi:.2f}%")
                # per-bagian
                by_bagian = df_sub.groupby('bagian_infer')['benar'].agg(['sum','count']).reset_index()
                for _, r in by_bagian.iterrows():
                    b = r['bagian_infer']; bn = int(r['sum']); ct = int(r['count']); sl = ct - bn
                    acc_b = 100.0 * bn / max(1, ct)
                    ringkas.append(f"  - {b.capitalize()}: N={ct} | Correct={bn} | Incorrect={sl} | Accuracy={acc_b:.2f}%")
            else:
                ringkas.append("Evaluasi: tidak ada GT di nama file (cari kata sehat/ringan/berat)." )
        except Exception:
            ringkas.append("Evaluasi: gagal menghitung metrik akurasi.")

        ringkas.append("Catatan: Bagian diambil dari bucket (Leaf/Frond/Root), bukan dari nama file. Jika semua kolom hilang (no overlap), file akan ditandai gagal.")

        self.pred_result_box.configure(state='normal')
        self.pred_result_box.delete('1.0', tk.END)
        self.pred_result_box.insert(tk.END, "\n".join(ringkas))
        self.pred_result_box.configure(state='disabled')

        try:
            base_dir = os.path.dirname(items[0][0]) if items else os.getcwd()
            df_out = pd.DataFrame(rows)
            kolom = ['filename', 'bagian_infer', 'label_prediksi_asli', 'kelas_pred', 'probabilitas', 'gt_from_filename']
            df_out = df_out.reindex(columns=[c for c in kolom if c in df_out.columns])
            self.last_pred_df = df_out
            self.last_pred_dir = base_dir
            self.btn_save_pred.config(state='normal')
            messagebox.showinfo(
                "Prediction Finished",
                "Prediksi selesai.\nKlik 'Save Results…' untuk menyimpan ringkasan ke file Excel."
            )
        except Exception as e:
            messagebox.showwarning("Gagal Menyiapkan Hasil", f"Prediction finished but results are not available: {e}")

    def save_pred_results(self):
        if getattr(self, "last_pred_df", None) is None or self.last_pred_df.empty:
            messagebox.showwarning("Tidak Ada Data", "There are no prediction results to save yet.")
            return

        initialdir = getattr(self, "last_pred_dir", None)
        if not initialdir or not os.path.isdir(initialdir):
            initialdir = os.getcwd()
        default_name = "prediksi_summary_selected_files_perbagian_9features.xlsx"

        path = filedialog.asksaveasfilename(
            title="Save Prediction Summary",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            initialdir=initialdir,
            initialfile=default_name
        )
        if not path:
            return

        try:
            lp = self.last_pred_df.copy()
            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                lp.to_excel(writer, index=False, sheet_name='prediksi')
            messagebox.showinfo("Tersimpan", f"Prediction summary saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Gagal Menyimpan", f"Gagal menyimpan ringkasan: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MLXGBGUI(root)
    root.mainloop()