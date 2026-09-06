"""
PD Detect — Model Trainer v3
Strategy: Train DIRECTLY on UCI clinical features (22 features, ground truth).
At inference, map extracted audio features → UCI-calibrated space via map_to_uci_vector().

Why this is better than synthetic audio:
- UCI features ARE the scientific gold standard for PD voice biomarkers
- No domain gap: model sees exactly the same 22-feature space at train & infer
- More stable: not affected by audio codec or microphone differences
- Data augmentation adds diversity without distorting clinical distributions

UCI dataset: Little et al. 2007, 195 samples (147 PD, 48 Healthy)
With augmentation + SMOTE-style oversampling → balanced 700+ training set
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import joblib

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_URL   = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"

# ── 22 UCI feature columns (in the exact order used by map_to_uci_vector) ──
UCI_FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR",
    "RPDE", "DFA",
    "spread1", "spread2", "D2", "PPE",
]


def augment_uci(df, noise_scale=0.08, seed=42):
    """
    Augment UCI data by adding small Gaussian noise (+-4% of std per feature).
    Target: exactly 50/50 class balance so no model needs pos_weight correction.
    """
    rng = np.random.default_rng(seed)
    feat_df = df[UCI_FEATURES].values.astype(np.float64)
    labels  = df["status"].values

    stds = np.clip(feat_df.std(axis=0), 1e-9, None)

    pd_idx = np.where(labels == 1)[0]   # 147 PD
    he_idx = np.where(labels == 0)[0]   #  48 Healthy

    # Augment both classes to exactly 1500 samples each
    target_each = 1500

    def oversample(idx, target, label_val):
        base_X = feat_df[idx]
        base_y = labels[idx]
        Xs = [base_X]
        ys = [base_y]
        needed = target - len(idx)
        while needed > 0:
            take = min(needed, len(idx))
            chosen = rng.choice(len(idx), take, replace=True)
            noise = rng.normal(0, noise_scale, (take, feat_df.shape[1])) * stds
            Xs.append(base_X[chosen] + noise)
            ys.append(np.full(take, label_val))
            needed -= take
        return np.vstack(Xs), np.concatenate(ys)

    pd_X, pd_y = oversample(pd_idx, target_each, 1)
    he_X, he_y = oversample(he_idx, target_each, 0)

    X = np.vstack([pd_X, he_X])
    y = np.concatenate([pd_y, he_y])
    print(f"  After augmentation: {len(X)} samples | PD: {sum(y==1)} | Healthy: {sum(y==0)}")
    return X, y


def train(X, y):
    # Classes are 50/50 balanced after augmentation — no weights needed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"  Train PD: {sum(y_train==1)} | Train Healthy: {sum(y_train==0)}")

    scaler     = StandardScaler()
    Xtr_sc     = scaler.fit_transform(X_train)
    Xte_sc     = scaler.transform(X_test)

    # ── 1. Random Forest (no class_weight needed — balanced data) ─────────
    print("\n[1/4] Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=7, min_samples_leaf=8,
        random_state=42, n_jobs=-1
    )
    rf.fit(Xtr_sc, y_train)
    _report("RF", rf, Xte_sc, y_test)

    # ── 2. XGBoost (scale_pos_weight=1 — balanced data) ───────────────────
    print("[2/4] XGBoost...")
    xgb_m = xgb.XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.75, colsample_bytree=0.75,
        scale_pos_weight=1.0,
        eval_metric="logloss", random_state=42, verbosity=0,
        reg_alpha=0.3, reg_lambda=1.5,
        min_child_weight=8
    )
    xgb_m.fit(Xtr_sc, y_train)
    _report("XGB", xgb_m, Xte_sc, y_test)

    # ── 3. SVM with Platt calibration ────────────────────────────────────
    print("[3/4] SVM (calibrated)...")
    svm = CalibratedClassifierCV(
        SVC(kernel="rbf", C=5, gamma="scale", random_state=42),
        cv=5, method="sigmoid"  # Platt scaling — better calibrated probabilities
    )
    svm.fit(Xtr_sc, y_train)
    _report("SVM", svm, Xte_sc, y_test)

    # ── 4. Gradient Boosting ──────────────────────────────────────────────
    print("[4/4] Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.08,
        subsample=0.8, random_state=42
    )
    gb.fit(Xtr_sc, y_train)
    _report("GB", gb, Xte_sc, y_test)

    # ── Ensemble ──────────────────────────────────────────────────────────
    print("\nEnsemble (soft voting: RF+XGB+SVM+GB)...")
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb_m), ("svm", svm), ("gb", gb)],
        voting="soft", weights=[2, 2, 1, 1]
    )
    ensemble.fit(Xtr_sc, y_train)

    ens_pred  = ensemble.predict(Xte_sc)
    ens_proba = ensemble.predict_proba(Xte_sc)[:, 1]
    ens_acc   = accuracy_score(y_test, ens_pred)
    ens_auc   = roc_auc_score(y_test, ens_proba)
    print(f"  Accuracy: {ens_acc:.4f}  |  AUC-ROC: {ens_auc:.4f}")
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, ens_pred))
    print(classification_report(y_test, ens_pred,
          target_names=["Healthy", "PD"], zero_division=0))

    # Cross-validation on training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_sc = cross_val_score(ensemble, Xtr_sc, y_train, cv=cv, scoring="balanced_accuracy")
    print(f"  5-fold balanced CV: {cv_sc.mean():.4f} ± {cv_sc.std():.4f}")

    # ── Post-training calibration check ────────────────────────────────
    # Verify that UCI healthy means → Healthy, UCI PD means → PD
    healthy_uci = np.array([181.94, 233.0, 136.5,
        0.00387, 2.3e-5, 0.00193, 0.00206, 0.00578,
        0.01762, 0.16296, 0.00950, 0.01051, 0.01330, 0.02851,
        0.01148, 24.679, 0.44255, 0.69572, -6.7593, 0.16029, 2.1545, 0.12302]).reshape(1,-1)
    pd_uci = np.array([145.18, 188.44, 108.89,
        0.00699, 5.07e-5, 0.00376, 0.00390, 0.01127,
        0.03366, 0.32120, 0.01768, 0.02028, 0.02760, 0.05303,
        0.02921, 20.974, 0.51682, 0.72541, -5.3334, 0.24813, 2.4561, 0.23383]).reshape(1,-1)
    h_p = float(ensemble.predict_proba(scaler.transform(healthy_uci))[0][1])
    pd_p = float(ensemble.predict_proba(scaler.transform(pd_uci))[0][1])
    print(f"\n  [CALIBRATION CHECK]")
    print(f"  UCI Healthy means -> PD prob: {h_p:.4f}  -> {'PASS (Healthy)' if h_p < 0.5 else 'FAIL (predicted PD!)'}")
    print(f"  UCI PD means      -> PD prob: {pd_p:.4f}  -> {'PASS (PD)' if pd_p >= 0.5 else 'FAIL (predicted Healthy!)'}")

    # Feature importances (RF + XGB combined)
    fi_rf  = rf.feature_importances_
    fi_xgb = xgb_m.feature_importances_
    fi_combined = (fi_rf + fi_xgb) / 2.0
    fi = dict(sorted(
        {n: float(v) for n, v in zip(UCI_FEATURES, fi_combined)}.items(),
        key=lambda x: x[1], reverse=True
    ))
    print("\nTop 10 UCI Features:")
    for i, (n, v) in enumerate(list(fi.items())[:10]):
        print(f"  {i+1:2d}. {n:<35s} {v:.4f}")

    return scaler, rf, xgb_m, svm, gb, ensemble, fi


def _report(name, model, X_test, y_test):
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    print(f"  {name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds,
          target_names=["Healthy", "PD"], zero_division=0))


def main():
    print("=" * 65)
    print("  PD Detect — Model Trainer v3 (UCI Feature Domain)")
    print("=" * 65)
    print()
    print("Strategy: Train DIRECTLY on UCI 22 clinical features.")
    print("  → Zero domain gap between training features and infer features")
    print("  → Augment + oversample healthy class for class balance")
    print()

    # ── Load UCI data ─────────────────────────────────────────────────────
    print("Downloading UCI Parkinson's dataset...")
    df = pd.read_csv(DATA_URL)
    df.drop(columns=["name"], inplace=True)
    n_pd  = sum(df["status"] == 1)
    n_he  = sum(df["status"] == 0)
    print(f"  {len(df)} samples | PD: {n_pd} | Healthy: {n_he}")
    print()

    # ── Augment to balanced dataset ───────────────────────────────────────
    print("Augmenting and balancing dataset...")
    X, y = augment_uci(df, noise_scale=0.05)
    print()

    # ── Train ─────────────────────────────────────────────────────────────
    print("Training models...")
    scaler, rf, xgb_m, svm, gb, ensemble, fi = train(X, y)

    # ── Save models ───────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler,   os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(rf,       os.path.join(MODELS_DIR, "random_forest.pkl"))
    joblib.dump(xgb_m,    os.path.join(MODELS_DIR, "xgboost.pkl"))
    joblib.dump(svm,      os.path.join(MODELS_DIR, "svm.pkl"))
    joblib.dump(gb,       os.path.join(MODELS_DIR, "gradient_boost.pkl"))
    joblib.dump(ensemble, os.path.join(MODELS_DIR, "ensemble.pkl"))

    # Remove feature_names.json → forces main.py to use UCI vector path
    fn_path = os.path.join(MODELS_DIR, "feature_names.json")
    if os.path.exists(fn_path):
        os.remove(fn_path)
        print("  Removed feature_names.json → inference uses UCI-mapped 22 features.")

    # Save feature importances
    with open(os.path.join(MODELS_DIR, "feature_importance.json"), "w") as f:
        json.dump({k: {"importance": v} for k, v in fi.items()}, f, indent=2)

    # Save UCI stats for calibration reference
    stats = {}
    feat_df = df[UCI_FEATURES]
    pd_df   = df[df["status"] == 1][UCI_FEATURES]
    he_df   = df[df["status"] == 0][UCI_FEATURES]
    for col in UCI_FEATURES:
        stats[col] = {
            "mean":         float(feat_df[col].mean()),
            "std":          float(feat_df[col].std()),
            "min":          float(feat_df[col].min()),
            "max":          float(feat_df[col].max()),
            "healthy_mean": float(he_df[col].mean()),
            "pd_mean":      float(pd_df[col].mean()),
        }
    with open(os.path.join(MODELS_DIR, "uci_feature_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nAll models saved to: {MODELS_DIR}")
    print("Training complete!")


if __name__ == "__main__":
    main()
