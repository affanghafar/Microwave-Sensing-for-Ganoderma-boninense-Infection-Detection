"""
XGBoost RADAR (Single Model) + GUI
=================================

Purpose:
- Train 1 XGBoost model for radar classification (classes: healthy/mild/severe)
- Radar features used: statistics for MPF and Mean Power (upper & lower)
- Supports:
  (A) Training no-CV: Early Stopping (ES) to find best iteration, then refit on full train.
  (B) Training CV: GridSearchCV (optional) + ES, then refit on full train (80%).

Important notes on results:
- This version only cleans up the structure + adds step-by-step comments.
- random_state/split/default parameter values are preserved so training results remain the same.

Expected dataset (Excel .xlsx):
- Required column: 'label'
- Numeric features: columns starting with prefixes in RADAR_FEATURE_PREFIXES
- Metadata columns to ignore: METADATA_COLS

Scotty — Luther project
"""


# === Global font 10pt + centered "Confusion Matrix" title for heatmaps ===
try:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 12,
        "figure.titlesize": 12,
    })
    try:
        import seaborn as sns
        if hasattr(sns, "heatmap"):
            _sns_heatmap_orig = sns.heatmap
            def _sns_heatmap_with_title(*args, **kwargs):
                ax = kwargs.get("ax", None)
                # Bold + resize ONLY the annotation numbers (does not affect axis tick labels)
                if kwargs.get("annot", False):
                    ak = dict(kwargs.get("annot_kws", {}) or {})
                    ak.setdefault("fontweight", "bold")
                    ak.setdefault("fontsize", 16)   # <- set number font size here
                    kwargs["annot_kws"] = ak
                out = _sns_heatmap_orig(*args, **kwargs)
                try:
                    _ax = ax if ax is not None else plt.gca()
                    _ax.text(0.5, 1.07, "Confusion Matrix",
                             transform=_ax.transAxes, ha="center", va="bottom",
                             fontsize=18, fontweight="bold")
                except Exception:
                    pass
                return out
            sns.heatmap = _sns_heatmap_with_title
    except Exception:
        pass
except Exception:
    pass

# XGboost_Radar_Single_Model.py
# =============================================================
#  RADAR MODEL - XGBoost with Early Stopping (no CV)
#  Dataset: Radar signal features (MPF, Mean Power)
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


# ==========================
#  GLOBAL CONSTANTS
# ==========================
RANDOM_SEED = 42
TEST_SIZE = 0.20          # outer split: train/test
ES_VALID_SIZE = 0.20      # inner split: train_in/valid for early stopping
EARLY_STOPPING_ROUNDS = 100
DEFAULT_N_ESTIMATORS_CAP = 1500

# --- UI label helpers ---
def _labels_to_english(seq):
    """Convert internal labels (healthy/mild/severe) to display labels (Healthy/Mild/Severe)."""
    mapping = {
        'healthy': 'Healthy',
        'mild':    'Mild',
        'severe':  'Severe',
    }
    out = []
    for x in seq:
        key = str(x).strip().lower()
        out.append(mapping.get(key, str(x).capitalize()))
    return out

try:
    from xgboost import XGBClassifier
    xgb_ready = True
except Exception:
    xgb_ready = False

# ==========================
#  HYPERPARAMETER CONFIG
# ==========================
HP_CONFIG = {
    "radar": {
        "learning_rate":    0.07,
        "max_depth":        4,
        "min_child_weight": 1,
        "subsample":        0.8,
        "colsample_bytree": 0.7,
        "gamma":            0,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "n_estimators":     1500,
        "use_gpu":          False,

        # --- Hyperparameter tuning (optional) ---
        # If enabled, GridSearchCV selects best_params, then we refit on full train.
        "use_cv":           False,
        "cv_splits":        3,
        "param_grid": {
            "learning_rate":    [0.05,0.06, 0.07],
            "max_depth":        [3, 4],
            "min_child_weight": [1, 2],
            "subsample":        [0.70, 0.80],
            "colsample_bytree": [0.60, 0.70],
            "gamma":            [0.0, 0.5],
            "reg_alpha":        [0.0, 0.1],
            "reg_lambda":       [1.0, 2.0],
        },
    }
}
# ==========================
#  UTILITIES
# ==========================

def sanitize_features(X: pd.DataFrame) -> pd.DataFrame:
    """Clean features: replace inf, convert to numeric, drop all-NaN columns"""
    X = X.copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.dropna(axis=1, how='all')
    return X


def simplify_kelas_from_label(label: str) -> str:
    """Extract class label (healthy/mild/severe) from string"""
    if not isinstance(label, str):
        return str(label)
    parts = label.lower().split('_')
    last = parts[-1] if parts else label.lower()
    if last in ('healthy', 'mild', 'severe'):
        return last
    # Also accept legacy Indonesian labels and map them
    if last == 'sehat' or 'sehat' in label.lower():
        return 'healthy'
    if last == 'ringan' or 'ringan' in label.lower():
        return 'mild'
    if last == 'berat' or 'berat' in label.lower():
        return 'severe'
    if 'severe' in label.lower():
        return 'severe'
    if 'mild' in label.lower():
        return 'mild'
    if 'healthy' in label.lower():
        return 'healthy'
    return last


def infer_gt_from_filename(fname: str):
    """Infer ground-truth from filename"""
    if not isinstance(fname, str):
        return None
    low = fname.lower()
    # English labels
    if 'healthy' in low: return 'healthy'
    if 'mild'    in low: return 'mild'
    if 'severe'  in low: return 'severe'
    # Legacy Indonesian labels
    if 'sehat'   in low: return 'healthy'
    if 'ringan'  in low: return 'mild'
    if 'berat'   in low: return 'severe'
    return None


# ==========================
#  RADAR FEATURE PREFIXES
# ==========================
# Features from radar dataset: MPF and Mean Power statistics
RADAR_FEATURE_PREFIXES = [
    "MPF atas (Hz)",
    "MPF bawah (Hz)", 
    "Mean Power atas (dB)",
    "Mean Power bawah (dB)"
]

# Metadata columns to exclude from features
METADATA_COLS = ['filename', 'label', 'idx', 'segment', 'n_points', 'time_start', 'time_end']


def prepare_xy_radar(df: pd.DataFrame):
    """Prepare X (features) and y (labels) from radar dataset"""
    if 'label' not in df.columns:
        return None, None, "Dataset does not contain a 'label' column."

    # Get all feature columns (exclude metadata)
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    
    # Filter to only radar feature columns that match our prefixes
    radar_cols = [c for c in feature_cols if any(c.startswith(p) for p in RADAR_FEATURE_PREFIXES)]
    
    if not radar_cols:
        return None, None, "No valid radar feature columns found (MPF/Mean Power)."

    X = df[radar_cols].copy()
    y = df['label'].astype(str).map(simplify_kelas_from_label)
    X = sanitize_features(X)
    
    return X, y, None



def prepare_X_radar_only(df: pd.DataFrame):
    """Prepare X (features) for radar prediction files (label optional).
    This is used for Batch Prediction on new/unlabeled data.
    """
    # Get all feature columns (exclude metadata)
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]

    # Filter to only radar feature columns that match our prefixes
    radar_cols = [c for c in feature_cols if any(c.startswith(p) for p in RADAR_FEATURE_PREFIXES)]

    if not radar_cols:
        return None, "No valid radar feature columns found (MPF/Mean Power)."

    X = df[radar_cols].copy()
    X = sanitize_features(X)
    return X, None

# ==========================
#  TRAINING FUNCTION
# ==========================

def run_xgb_training_radar_no_cv(data_path: str, hp: dict):
    """Train XGBoost model on radar dataset with early stopping"""

    # --------------------------
    # STEP-BY-STEP (No-CV + ES)
    # 1) Load Excel dataset
    # 2) Extract X (features) and y (labels), clean NaN/inf
    # 3) Encode labels -> integer (LabelEncoder)
    # 4) Outer split: train/test (stratified)
    # 5) Inner split in train: train_in/valid (for Early Stopping)
    # 6) Impute missing values (median) for ES stage
    # 7) Train model with ES to find best_iteration + history metrics
    # 8) Refit model on full train (80%) with n_estimators = best_iter+1
    # 9) Evaluate on test set: Accuracy, Macro-F1, Report, Confusion Matrix
    # 10) (Optional) compute SHAP mean(|SHAP|) for feature interpretation
    # --------------------------
    if not xgb_ready:
        return "XGBoost is not installed. Install with: pip install xgboost"

    # Load data
    try:
        data = pd.read_excel(data_path)
    except Exception as e:
        return f"Error reading Excel file: {e}"

    X, y, err = prepare_xy_radar(data)
    if err:
        return err

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Train-test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, stratify=y_enc, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )
    except ValueError as e:
        return f"[RADAR] Error in train-test split: {e}"

    # Inner split for early stopping validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, stratify=y_train, test_size=ES_VALID_SIZE, random_state=RANDOM_SEED
    )

    # Impute missing values
    imputer_es = SimpleImputer(strategy="median")
    X_tr_imp  = imputer_es.fit_transform(X_tr)
    X_val_imp = imputer_es.transform(X_val)

    device = "cuda" if bool(hp.get("use_gpu", False)) else "cpu"
    n_estimators_cap = int(hp.get("n_estimators", DEFAULT_N_ESTIMATORS_CAP))

    # Train with early stopping to find best iteration
    model_es = XGBClassifier(
        booster="gbtree",
        objective="multi:softprob",
        n_estimators=n_estimators_cap,
        tree_method="hist",
        eval_metric=["merror","mlogloss"],
        random_state=RANDOM_SEED,
        n_jobs=-1,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
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
        if len(arr_merror) > best_iter:
            best_merror = arr_merror[best_iter]
        if len(arr_mlogloss) > best_iter:
            best_mlogloss = arr_mlogloss[best_iter]
    else:
        best_iter = min(300, n_estimators_cap)

    # Refit on full training set with best iteration
    imputer_full = SimpleImputer(strategy="median")
    X_train_imp  = imputer_full.fit_transform(X_train)
    X_test_imp   = imputer_full.transform(X_test)

    model_refit = XGBClassifier(
        booster="gbtree",
        objective="multi:softprob",
        n_estimators=int(best_iter) + 1,
        tree_method="hist",
        random_state=RANDOM_SEED,
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

    # Evaluate on test set
    y_pred = model_refit.predict(X_test_imp)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    y_test_label = le.inverse_transform(y_test)
    y_pred_label = le.inverse_transform(y_pred)
    report = classification_report(y_test_label, y_pred_label, zero_division=0, digits=4)
    cm = confusion_matrix(y_test_label, y_pred_label, labels=le.classes_)

    # SHAP feature importance
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

    # Visualizations
    try:
        if shap_top10_pairs is not None:
            names = [a for a,_ in shap_top10_pairs]
            vals  = [float(b) for _,b in shap_top10_pairs]
            yy = np.arange(len(names))
            plt.figure(figsize=(8.2, 5.6))
            plt.barh(yy, vals)
            plt.yticks(yy, names, fontsize=9)
            plt.gca().invert_yaxis()
            plt.xlabel("mean(|SHAP|)")
            plt.title("SHAP Feature Importance — RADAR Model")
            plt.tight_layout()
            plt.show()
    except Exception as _e:
        print("SHAP barplot failed:", _e)

    # Confusion matrix
    plt.figure(figsize=(6, 5))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=_labels_to_english(le.classes_), 
                yticklabels=_labels_to_english(le.classes_))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

    # Training history
    if arr_merror and arr_mlogloss:
        iters = np.arange(len(arr_mlogloss))
        plt.figure(figsize=(7.2, 4.2))
        plt.plot(iters, arr_merror, label='merror')
        plt.plot(iters, arr_mlogloss, label='mlogloss')
        if best_iter is not None:
            plt.axvline(best_iter, linestyle='--', alpha=0.6, label=f'best_iter={best_iter}')
        plt.xlabel('Iteration')
        plt.ylabel('Metric value')
        plt.title('Validation Metrics — RADAR Model')
        plt.legend()
        plt.tight_layout()
        plt.show()

    result_str = (
        f"=== [RADAR] XGBoost + Early Stopping ===\n"
        f"Dataset: {os.path.basename(data_path)}\n"
        f"Training samples: {len(X_train)} | Test samples: {len(X_test)}\n"
        f"Features used: {len(X.columns)} radar features\n"
        f"Classes: {list(le.classes_)}\n\n"
        f"n_estimators upper bound: {n_estimators_cap}\n"
        f"Device: {'GPU (cuda)' if hp.get('use_gpu', False) else 'CPU'}\n"
        f"Hyperparameters: {', '.join(f'{k}={hp[k]}' for k in ['learning_rate','max_depth','min_child_weight','subsample','colsample_bytree','gamma'])}\n\n"
        f"Best Iteration (ES): {best_iter}\n"
        f"Validation merror @best: {best_merror}\n"
        f"Validation mlogloss @best: {best_mlogloss}\n\n"
        f"Test Accuracy: {acc:.4f}\n"
        f"Test F1-macro: {f1m:.4f}\n\n"
        "Classification Report (Test):\n"
        f"{report}\n"
        "Confusion Matrix:\n"
        f"{cm}\n\n"
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
        "hp": hp,
        "n_estimators_upper": n_estimators_cap,
        "shap_mean_abs": dict(shap_pairs) if shap_pairs else None,
        "shap_top10": shap_top10_pairs,
    }

    return result_str, artifacts


# ==========================
#  GUI
# ==========================



def run_xgb_training_radar_cv_refit80(data_path: str, hp: dict):
    """Train XGBoost model on radar dataset with:
    - 80/20 train-test split
    - internal 80/20 split inside train for early stopping (≈64/16/20)
    - GridSearchCV over a small param grid (dict-of-lists)
    - FINAL refit on full 80% train using best params + best iteration

    This keeps the testing set untouched during hyperparameter search.
    """

    # --------------------------
    # STEP-BY-STEP (CV + ES + Refit80)
    # 1) Load Excel dataset
    # 2) Extract X and y, clean NaN/inf
    # 3) Encode labels -> integer
    # 4) Outer split: train/test (stratified) => test set is "sterile" (not touched by CV)
    # 5) Inner split in train: train_in/valid (for Early Stopping)
    # 6) Impute median on train_in and valid (ES stage)
    # 7) GridSearchCV on train_in (scoring=accuracy), ES monitored on valid
    # 8) Take best_params + best_iteration (from ES)
    # 9) FINAL refit on full train (80%) with n_estimators = best_iter+1
    # 10) Evaluate on test set + plot (CM, history, SHAP)
    #
    # Note:
    # - param_grid here still has "singleton" values (1 value per parameter),
    #   so CV does not actually sweep many combinations.
    #   To sweep according to HP_CONFIG["radar"]["param_grid"], replace
    #   param_grid = hp["param_grid"] (results may change if CV is enabled).
    # --------------------------
    if not xgb_ready:
        return "XGBoost is not installed. Install with: pip install xgboost"

    # Load data
    try:
        data = pd.read_excel(data_path)
    except Exception as e:
        return f"Error reading Excel file: {e}"

    X, y, err = prepare_xy_radar(data)
    if err:
        return err

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Train-test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, stratify=y_enc, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )
    except ValueError as e:
        return f"[RADAR] Error in train-test split: {e}"

    # Inner split for early stopping validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, stratify=y_train, test_size=ES_VALID_SIZE, random_state=RANDOM_SEED
    )

    # Impute missing values for ES stage
    imputer_es = SimpleImputer(strategy="median")
    X_tr_imp  = imputer_es.fit_transform(X_tr)
    X_val_imp = imputer_es.transform(X_val)

    device = "cuda" if bool(hp.get("use_gpu", False)) else "cpu"
    n_estimators_cap = int(hp.get("n_estimators", DEFAULT_N_ESTIMATORS_CAP))
    cv_splits = int(hp.get("cv_splits", 3))

    # Base model for grid search (early stopping enabled)
    base_xgb = XGBClassifier(
        booster="gbtree",
        objective="multi:softprob",
        n_estimators=n_estimators_cap,
        tree_method="hist",
        eval_metric=["merror","mlogloss"],
        random_state=RANDOM_SEED,
        n_jobs=-1,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        device=device,
        # defaults (overridden by GridSearchCV)
        learning_rate=float(hp.get("learning_rate", 0.06)),
        max_depth=int(hp.get("max_depth", 5)),
        min_child_weight=int(hp.get("min_child_weight", 1)),
        subsample=float(hp.get("subsample", 0.85)),
        colsample_bytree=float(hp.get("colsample_bytree", 0.7)),
        gamma=float(hp.get("gamma", 0.5)),
        reg_alpha=float(hp.get("reg_alpha", 0.0)),
        reg_lambda=float(hp.get("reg_lambda", 1.0)),
    )

    # === Param grid: style dict-of-lists (seperti contohmu) ===
    param_grid = {
        "learning_rate":    [0.07],
        "max_depth":        [4],
        "min_child_weight": [1],
        "subsample":        [0.8],
        "colsample_bytree": [0.7],
        "gamma":            [0.5],
        "reg_alpha":        [0.0],
        "reg_lambda":       [1.0],
    }

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_SEED)

    gs = GridSearchCV(
        estimator=base_xgb,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0
    )

    # Grid search on train_in; early stopping monitored on hold-out valid
    gs.fit(X_tr_imp, y_tr, eval_set=[(X_val_imp, y_val)], verbose=False)

    model_es = gs.best_estimator_
    best_params = gs.best_params_ if hasattr(gs, "best_params_") else {}

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
        if len(arr_merror) > best_iter:
            best_merror = arr_merror[best_iter]
        if len(arr_mlogloss) > best_iter:
            best_mlogloss = arr_mlogloss[best_iter]
    else:
        best_iter = min(300, n_estimators_cap)

    # === FINAL REFIT on full 80% train ===
    imputer_full = SimpleImputer(strategy="median")
    X_train_imp  = imputer_full.fit_transform(X_train)
    X_test_imp   = imputer_full.transform(X_test)

    model_refit = XGBClassifier(
        booster="gbtree",
        objective="multi:softprob",
        n_estimators=int(best_iter) + 1,
        tree_method="hist",
        eval_metric=["merror","mlogloss"],
        random_state=RANDOM_SEED,
        n_jobs=-1,
        device=device,
        **best_params
    )
    model_refit.fit(X_train_imp, y_train)

    # Evaluate on test set
    y_pred = model_refit.predict(X_test_imp)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    y_test_label = le.inverse_transform(y_test)
    y_pred_label = le.inverse_transform(y_pred)
    report = classification_report(y_test_label, y_pred_label, zero_division=0, digits=4)
    cm = confusion_matrix(y_test_label, y_pred_label, labels=le.classes_)

    # SHAP feature importance (best-effort)
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

    # SHAP barplot
    try:
        if shap_top10_pairs is not None:
            names = [a for a,_ in shap_top10_pairs]
            vals  = [float(b) for _,b in shap_top10_pairs]
            yy = np.arange(len(names))
            plt.figure(figsize=(8.2, 5.6))
            plt.barh(yy, vals)
            plt.yticks(yy, names, fontsize=9)
            plt.gca().invert_yaxis()
            plt.xlabel("mean(|SHAP|)")
            plt.title("SHAP Feature Importance — RADAR Model (CV+Refit80)")
            plt.tight_layout()
            plt.show()
    except Exception:
        pass

    # Confusion matrix plot
    try:
        import seaborn as sns
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
                    xticklabels=_labels_to_english(le.classes_),
                    yticklabels=_labels_to_english(le.classes_))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()
    except Exception:
        pass

    # Training history
    if arr_merror and arr_mlogloss:
        iters = np.arange(len(arr_mlogloss))
        plt.figure(figsize=(7.2, 4.2))
        plt.plot(iters, arr_merror, label='merror')
        plt.plot(iters, arr_mlogloss, label='mlogloss')
        if best_iter is not None:
            plt.axvline(best_iter, linestyle='--', alpha=0.6, label=f'best_iter={best_iter}')
        plt.xlabel('Iteration')
        plt.ylabel('Metric value')
        plt.title('Validation Metrics — RADAR Model (CV Search)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Summary text (simplified to avoid key errors)
    result_str = (
        "=== [RADAR] XGBoost + GridSearchCV + Refit 80/20 ===\n"
        f"Dataset: {os.path.basename(data_path)}\n"
        f"Training samples: {len(X_train)} | Test samples: {len(X_test)}\n"
        f"Internal split: train_in={len(X_tr)} | valid={len(X_val)}\n"
        f"Features used: {len(X.columns)} radar features\n"
        f"Classes: {list(le.classes_)}\n\n"
        f"CV folds: {cv_splits}\n"
        f"Total combinations: {len(param_grid['learning_rate']) * len(param_grid['max_depth']) * len(param_grid['min_child_weight']) * len(param_grid['subsample']) * len(param_grid['colsample_bytree']) * len(param_grid['gamma']) * len(param_grid['reg_alpha']) * len(param_grid['reg_lambda'])}\n"
        f"n_estimators upper bound: {n_estimators_cap}\n"
        f"Device: {'GPU (cuda)' if hp.get('use_gpu', False) else 'CPU'}\n\n"
        f"Best Params (CV): {best_params}\n"
        f"Best Iteration (ES): {best_iter}\n"
        f"Validation merror @best: {best_merror}\n"
        f"Validation mlogloss @best: {best_mlogloss}\n\n"
        f"Test Accuracy: {acc:.4f}\n"
        f"Test F1-macro: {f1m:.4f}\n\n"
        "Classification Report (Test):\n"
        f"{report}\n"
        "Confusion Matrix:\n"
        f"{cm}\n\n"
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
        "hp": dict(hp, **{"best_params": best_params}),
        "n_estimators_upper": n_estimators_cap,
        "shap_mean_abs": dict(shap_pairs) if shap_pairs else None,
        "shap_top10": shap_top10_pairs,
    }

    return result_str, artifacts

#  GUI
# ==========================



class MLXGBRadarGUI:
    def __init__(self, master):
        self.master = master
        master.title("Radar Signal Analyzer — XGBoost Classification")
        master.geometry("1180x780")
        master.minsize(1040, 700)

        self.model_radar = None
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
        nb.add(self.tab_ml, text='Training & Prediction')

        self.build_ml_tab()

    def build_ml_tab(self):
        # 1) Training file input
        s1 = ttk.LabelFrame(self.tab_ml, text="1. Training Data (Excel file with radar features)", padding=(8,4))
        s1.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(s1, text="Training Dataset (.xlsx):").grid(row=0, column=0, padx=4, sticky='e')
        self.mlfile_var = tk.StringVar()
        ttk.Entry(s1, textvariable=self.mlfile_var, width=78).grid(row=0, column=1, padx=4)
        ttk.Button(s1, text="Browse", command=self.browse_mlfile).grid(row=0, column=2, padx=4)

        # 2) Training controls
        s2 = ttk.LabelFrame(self.tab_ml, text="2. Model Training & Evaluation", padding=(8,4))
        s2.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        topbar = ttk.Frame(s2)
        topbar.pack(fill=tk.X, pady=(4,6))
        ttk.Button(topbar, text="Train RADAR Model", command=self.run_ml_radar).pack(side=tk.LEFT, padx=4)
        # CV hyperparameter tuning controls
        self.use_cv_var = tk.BooleanVar(value=bool(HP_CONFIG["radar"].get("use_cv", False)))
        self.cv_splits_var = tk.IntVar(value=int(HP_CONFIG["radar"].get("cv_splits", 3)))
        ttk.Checkbutton(topbar, text="Enable CV tuning", variable=self.use_cv_var).pack(side=tk.LEFT, padx=(12,4))
        ttk.Label(topbar, text="CV folds:").pack(side=tk.LEFT)
        ttk.Spinbox(topbar, from_=2, to=10, width=3, textvariable=self.cv_splits_var).pack(side=tk.LEFT, padx=(4,12))
        ttk.Button(topbar, text="Save Model (.pkl)", command=self.save_model).pack(side=tk.LEFT, padx=4)
        ttk.Button(topbar, text="Load Model (.pkl)", command=self.load_model).pack(side=tk.LEFT, padx=4)

        self.result_txt = scrolledtext.ScrolledText(s2, width=120, height=18, font=('Consolas', 9), wrap='word')
        self.result_txt.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # 3) Prediction section
        s3 = ttk.LabelFrame(self.tab_ml, text="3. Batch Prediction (Excel files; label optional)", padding=(8,4))
        s3.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        lb = tk.Listbox(s3, selectmode=tk.EXTENDED, height=7)
        lb.grid(row=0, column=0, columnspan=2, sticky='nsew', padx=4, pady=4)
        sb = ttk.Scrollbar(s3, orient='vertical', command=lb.yview)
        sb.grid(row=0, column=2, sticky='ns')
        lb.config(yscrollcommand=sb.set)
        self.pred_listbox = lb

        btns = ttk.Frame(s3)
        btns.grid(row=1, column=0, columnspan=3, sticky='w', padx=4, pady=2)

        def add_files():
            paths = filedialog.askopenfilenames(
                title="Select Excel files for prediction",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            if not paths:
                return
            if isinstance(paths, str):
                paths = self.master.tk.splitlist(paths)
            existing = set(lb.get(0, tk.END))
            for p in paths:
                if p not in existing:
                    lb.insert(tk.END, p)

        def remove_sel():
            sel = list(lb.curselection())
            if not sel:
                messagebox.showinfo("Info", "Select items to remove (Ctrl/Shift for multi-select)")
                return
            for idx in reversed(sel):
                lb.delete(idx)

        def clear_all():
            lb.delete(0, tk.END)

        ttk.Button(btns, text="Add Files", command=add_files).grid(row=0, column=0, padx=4)
        ttk.Button(btns, text="Remove Selected", command=remove_sel).grid(row=0, column=1, padx=4)
        ttk.Button(btns, text="Clear All", command=clear_all).grid(row=0, column=2, padx=4)

        s3.grid_rowconfigure(0, weight=1)
        s3.grid_columnconfigure(0, weight=1)

        bottom = ttk.Frame(s3)
        bottom.grid(row=2, column=0, columnspan=3, sticky='w', padx=4, pady=6)
        ttk.Button(bottom, text="Predict All Files", command=self.run_predict_files).grid(row=0, column=0, padx=4)

        self.btn_save_pred = ttk.Button(bottom, text="Save Results", command=self.save_pred_results, state='disabled')
        self.btn_save_pred.grid(row=0, column=1, padx=8)

        self.pred_result_box = scrolledtext.ScrolledText(s3, width=120, height=12, font=('Consolas', 9), wrap='word')
        self.pred_result_box.grid(row=3, column=0, columnspan=3, sticky='nsew', padx=4, pady=4)
        self.pred_result_box.configure(state='disabled')

        s3.grid_rowconfigure(3, weight=1)

    def browse_mlfile(self):
        path = filedialog.askopenfilename(
            title="Select training dataset",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if path:
            self.mlfile_var.set(path)

    def run_ml_radar(self):
        data_path = self.mlfile_var.get()

        if not xgb_ready:
            messagebox.showerror("Error", "XGBoost not installed. Run: pip install xgboost")
            return
        if not data_path or not os.path.exists(data_path):
            messagebox.showerror("Error", "Please select a training dataset file!")
            return

        self.result_txt.delete("1.0", tk.END)
        self.result_txt.insert(tk.END, "Starting RADAR model training...\n\n")
        self.master.update()
        # Read HP config, then override from GUI toggles
        hp = dict(HP_CONFIG["radar"])
        try:
            hp["use_cv"] = bool(self.use_cv_var.get())
            hp["cv_splits"] = int(self.cv_splits_var.get())
        except Exception:
            pass

        try:
            res = run_xgb_training_radar_cv_refit80(data_path=data_path, hp=hp) if hp.get("use_cv", False) else run_xgb_training_radar_no_cv(data_path=data_path, hp=hp)
        except Exception as e:
            self.result_txt.insert(tk.END, f"Training FAILED: {e}\n\n")
            messagebox.showerror("Training Error", str(e))
            return

        if isinstance(res, str):
            self.result_txt.insert(tk.END, f"{res}\n\n")
            messagebox.showwarning("Training Issue", res)
            return

        summary_text, artifacts = res
        self.model_radar = artifacts
        self.last_summary_text = summary_text
        
        self.result_txt.delete("1.0", tk.END)
        self.result_txt.insert(tk.END, summary_text)
        messagebox.showinfo("Success", "Model trained successfully!")

    def save_model(self):
        if self.model_radar is None:
            messagebox.showwarning("No Model", "Train a model first.")
            return
        
        path = filedialog.asksaveasfilename(
            title="Save RADAR Model",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            initialfile="radar_model.pkl"
        )
        if not path:
            return
        
        try:
            with open(path, "wb") as f:
                pickle.dump({
                    "model": self.model_radar, 
                    "summary_text": self.last_summary_text
                }, f)
            messagebox.showinfo("Success", f"Model saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save model:\n{e}")

    def load_model(self):
        path = filedialog.askopenfilename(
            title="Load RADAR Model",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if not path:
            return
        
        try:
            with open(path, "rb") as f:
                bundle = pickle.load(f)

            md = bundle.get("model", None)
            if md is None:
                raise ValueError("Invalid model file")

            self.model_radar = md
            self.last_summary_text = bundle.get("summary_text", "")

            self.result_txt.delete("1.0", tk.END)
            if self.last_summary_text:
                self.result_txt.insert(tk.END, self.last_summary_text)
            else:
                self.result_txt.insert(tk.END, "Model loaded successfully.")

            # Show confusion matrix
            art = self.model_radar
            cm = art.get("confusion_matrix", None)
            classes = art.get("label_classes", None)
            if isinstance(cm, np.ndarray) and classes is not None:
                import seaborn as sns
                plt.figure(figsize=(6,5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
                           xticklabels=_labels_to_english(classes), 
                           yticklabels=_labels_to_english(classes))
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.tight_layout()
                plt.show()

            # Show training history
            ev = art.get("eval_history", None)
            bi = art.get("best_iter", None)
            if isinstance(ev, dict) and "validation_0" in ev:
                val_hist = ev["validation_0"]
                arr_merror = val_hist.get("merror", [])
                arr_mlogloss = val_hist.get("mlogloss", [])
                if arr_merror and arr_mlogloss:
                    iters = np.arange(len(arr_mlogloss))
                    plt.figure(figsize=(7.2, 4.2))
                    plt.plot(iters, arr_merror, label='merror')
                    plt.plot(iters, arr_mlogloss, label='mlogloss')
                    if bi is not None:
                        plt.axvline(bi, linestyle='--', alpha=0.6, label=f'best_iter={bi}')
                    plt.xlabel('Iteration')
                    plt.ylabel('Metric')
                    plt.title('Training History (Loaded)')
                    plt.legend()
                    plt.tight_layout()
                    plt.show()

            messagebox.showinfo("Success", "Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load model:\n{e}")

    def run_predict_files(self):

        # --------------------------
        # STEP-BY-STEP (Batch Prediction)
        # 1) Ensure model exists (train/load)
        # 2) Get list of Excel files from listbox
        # 3) For each file:
        #    a) Read Excel -> df_pred
        #    b) Get radar feature columns matching prefix (label optional)
        #    c) Align column order with training features (reindex)
        #    d) Impute missing values (use imputer from training)
        #    e) Predict class + probability
        #    f) Save per-row results for export
        # 4) Compute simple evaluation if GT is available (label column / filename)
        # 5) Display summary in textbox + prepare Excel export
        # --------------------------
        if self.model_radar is None:
            messagebox.showwarning("No Model", "Train or load a model first.")
            return

        files = list(self.pred_listbox.get(0, tk.END))
        if not files:
            messagebox.showwarning("No Files", "Add files for prediction first.")
            return

        self.last_pred_df = None
        self.last_pred_dir = None
        self.btn_save_pred.config(state='disabled')

        results, rows = [], []
        total, ok, fail = len(files), 0, 0
        art = self.model_radar

        for fp in files:
            try:
                if not os.path.exists(fp):
                    raise FileNotFoundError("File not found")

                # Load the prediction file
                df_pred = pd.read_excel(fp)
                
                # Prepare features
                X_pred, err = prepare_X_radar_only(df_pred)
                if err:
                    raise ValueError(err)

                # Ensure features match training
                feat_cols = art['feature_columns']
                X_pred = X_pred.reindex(columns=feat_cols, fill_value=0)
                X_pred = X_pred.astype('float64')
                
                # Impute and predict
                X_pred_imp = art['imputer'].transform(X_pred)
                model = art['model']
                le = art['label_encoder']

                y_pred_enc = model.predict(X_pred_imp)
                y_pred_labels = le.inverse_transform(y_pred_enc)
                
                # Get probabilities
                probas = model.predict_proba(X_pred_imp)
                
                # Process each row
                for i, (pred_label, proba_row) in enumerate(zip(y_pred_labels, probas)):
                    class_label = simplify_kelas_from_label(pred_label)
                    
                    # Get probability for predicted class
                    classes = list(le.classes_)
                    idx = classes.index(pred_label)
                    prob_val = float(proba_row[idx])
                    
                    # Get original filename if available
                    orig_filename = df_pred.iloc[i]['filename'] if 'filename' in df_pred.columns else f"row_{i}"
                    # Prefer GT from label column if present; fallback to filename inference
                    gt = None
                    if 'label' in df_pred.columns:
                        try:
                            gt = simplify_kelas_from_label(str(df_pred.iloc[i]['label']))
                        except Exception:
                            gt = None
                    if gt is None:
                        gt = infer_gt_from_filename(str(orig_filename))
                    
                    results.append(f"{orig_filename}  →  {class_label} ({prob_val:.3f})")
                    
                    rows.append({
                        'source_file': os.path.basename(fp),
                        'filename': orig_filename,
                        'label_pred': str(pred_label),
                        'class_pred': str(class_label),
                        'probability': prob_val,
                        'gt_from_filename': gt
                    })
                
                ok += 1

            except Exception as e:
                results.append(f"{os.path.basename(fp)}  →  FAILED: {e}")
                fail += 1

        # Summary statistics
        ringkas = [
            f"Batch Prediction Results",
            f"Total files: {total} | Success: {ok} | Failed: {fail}",
            f"Total predictions: {len(rows)}",
            "-"*70
        ] + results + ["-"*70]

        # Calculate accuracy from ground truth in filenames
        try:
            df_eval = pd.DataFrame(rows) if rows else pd.DataFrame()
            has_gt = df_eval['gt_from_filename'].notna().sum() if not df_eval.empty else 0
            
            if has_gt > 0:
                df_sub = df_eval.dropna(subset=['gt_from_filename']).copy()
                df_sub['correct'] = (df_sub['class_pred'].astype(str) == df_sub['gt_from_filename'].astype(str)).astype(int)
                correct = int(df_sub['correct'].sum())
                incorrect = int(len(df_sub) - correct)
                accuracy = 100.0 * correct / max(1, len(df_sub))
                
                ringkas.append(f"\nAccuracy Evaluation (from label column or filenames with healthy/mild/severe):")
                ringkas.append(f"  Samples: {len(df_sub)} | Correct: {correct} | Incorrect: {incorrect}")
                ringkas.append(f"  Accuracy: {accuracy:.2f}%")
                
                # Per-class breakdown
                class_stats = []
                for cls in ['healthy', 'mild', 'severe']:
                    df_class = df_sub[df_sub['gt_from_filename'] == cls]
                    if len(df_class) > 0:
                        class_correct = int(df_class['correct'].sum())
                        class_total = len(df_class)
                        class_acc = 100.0 * class_correct / class_total
                        class_stats.append(f"    {cls.capitalize()}: {class_correct}/{class_total} ({class_acc:.1f}%)")
                
                if class_stats:
                    ringkas.append("  Per-class accuracy:")
                    ringkas.extend(class_stats)
            else:
                ringkas.append("\nNo ground truth found in filenames (looking for healthy/mild/severe)")
        except Exception as e:
            ringkas.append(f"\nAccuracy calculation failed: {e}")

        # Display results
        self.pred_result_box.configure(state='normal')
        self.pred_result_box.delete('1.0', tk.END)
        self.pred_result_box.insert(tk.END, "\n".join(ringkas))
        self.pred_result_box.configure(state='disabled')

        # Save results to dataframe
        try:
            base_dir = os.path.dirname(files[0]) if files else os.getcwd()
            df_out = pd.DataFrame(rows)
            self.last_pred_df = df_out
            self.last_pred_dir = base_dir
            self.btn_save_pred.config(state='normal')
            messagebox.showinfo("Prediction Complete", 
                              f"Predicted {len(rows)} samples from {ok} files.\n\n"
                              "Click 'Save Results' to export to Excel.")
        except Exception as e:
            messagebox.showwarning("Results Ready", 
                                 f"Prediction complete but couldn't prepare export: {e}")

    def save_pred_results(self):
        if getattr(self, "last_pred_df", None) is None or self.last_pred_df.empty:
            messagebox.showwarning("No Data", "No prediction results to save.")
            return

        initialdir = getattr(self, "last_pred_dir", None) or os.getcwd()
        default_name = "radar_predictions.xlsx"

        path = filedialog.asksaveasfilename(
            title="Save Prediction Results",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            initialdir=initialdir,
            initialfile=default_name
        )
        if not path:
            return

        try:
            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                self.last_pred_df.to_excel(writer, index=False, sheet_name='predictions')
            messagebox.showinfo("Saved", f"Prediction results saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save results:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MLXGBRadarGUI(root)
    root.mainloop()