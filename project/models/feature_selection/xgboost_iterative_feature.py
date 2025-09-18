from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, precision_score, recall_score,
    precision_recall_curve, roc_curve, ConfusionMatrixDisplay
)
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import argparse, json
from joblib import load, dump
from pathlib import Path


def _cv_score_xgb(
        X, y, features, metric="average_precision",
        n_splits=5, random_state=42, xgb_params=None,
        early_stopping_rounds=50, groups=None
):
    es_metric = "aucpr" if metric == "average_precision" else ("auc" if metric == "roc_auc" else "logloss")
    pos_weight = len(y[y == 0]) / len(y[y == 1]) if y.sum() > 0 else 1

    xgb_params = xgb_params or dict(
        n_estimators=1000, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
        eval_metric=es_metric, reg_lambda=1.0
    )
    Xf = X[features].values
    if groups is None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = cv.split(Xf, y)
    else:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = cv.split(Xf, y, groups=groups)

    scores = []
    for tr_idx, va_idx in splits:
        X_tr, X_va = Xf[tr_idx], Xf[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        pos = y_tr.sum()
        neg = len(y_tr) - pos
        pos_weight_fold = (neg / pos) if pos > 0 else 1.0

        params = {**xgb_params, "scale_pos_weight": pos_weight_fold}
        model = XGBClassifier(**params, early_stopping_rounds=early_stopping_rounds, random_state=random_state)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False
        )
        y_score = model.predict_proba(X_va)[:, 1]
        if metric == "average_precision":
            s = average_precision_score(y_va, y_score)
        elif metric == "roc_auc":
            s = roc_auc_score(y_va, y_score)
        else:
            raise ValueError("metric must be 'average_precision' or 'roc_auc'")
        scores.append(s)

    return float(np.mean(scores)), float(np.std(scores))


def _prefilter_by_gain(X, y, top_k=200, random_state=42, xgb_params=None):
    pos_weight = len(y[y == 0]) / len(y[y == 1]) if y.sum() > 0 else 1

    xgb_params = xgb_params or dict(
        n_estimators=200, learning_rate=0.1, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
        eval_metric="logloss", reg_lambda=1.0, scale_pos_weight=pos_weight
    )
    model = XGBClassifier(**xgb_params, random_state=random_state)
    model.fit(X, y)
    gain = model.feature_importances_
    order = np.argsort(gain)[::-1]
    feats = list(X.columns[order][:min(top_k, X.shape[1])])
    return feats


def greedy_forward_select_xgb(
        X: pd.DataFrame,
        y: np.ndarray,
        metric: str = "average_precision",
        candidate_features: list | None = None,
        max_features: int | None = None,
        min_improvement: float = 1e-4,
        patience: int = 10,
        n_splits: int = 5,
        random_state: int = 42,
        groups=None
):
    if candidate_features is None:
        candidate_features = _prefilter_by_gain(X, y, top_k=200, random_state=random_state)
        print(f"[prefilter] Using top {len(candidate_features)} gain features as candidates.")
    else:
        candidate_features = [f for f in candidate_features if f in X.columns]

    remaining = set(candidate_features)
    selected = []
    best_first, best_first_score = None, -np.inf

    # Start with best single feature
    for f in tqdm(remaining, desc="Single-feature scan", leave=False):
        s, _ = _cv_score_xgb(X, y, [f], metric=metric, n_splits=n_splits,
                             random_state=random_state, groups=groups)
        if s > best_first_score:
            best_first, best_first_score = f, s
    selected.append(best_first)
    remaining.remove(best_first)
    best_score = best_first_score
    history = [(len(selected), best_score, best_first)]
    print(f"[start] 1 feature: {best_first} → {best_score:.4f} ({metric})")

    # Greedy add
    no_improve_rounds = 0
    while remaining and (max_features is None or len(selected) < max_features):
        best_candidate, best_candidate_score = None, best_score
        for f in tqdm(list(remaining), desc=f"Adding feature {len(selected) + 1}", leave=False):
            trial_feats = selected + [f]
            s, _ = _cv_score_xgb(X, y, trial_feats, metric=metric, n_splits=n_splits,
                                 random_state=random_state, groups=groups)
            if s > best_candidate_score:
                best_candidate, best_candidate_score = f, s

        if best_candidate is not None and (best_candidate_score - best_score) >= min_improvement:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            best_score = best_candidate_score
            history.append((len(selected), best_score, best_candidate))
            print(f"[add] {len(selected)} feats: +{best_candidate} → {best_score:.4f}")
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1
            print(f"[no improve] rounds={no_improve_rounds} (best={best_score:.4f})")
            if no_improve_rounds >= patience:
                print("[stop] early stop by patience.")
                break

    return selected, pd.DataFrame(history, columns=["k_features", f"cv_{metric}", "last_added"])


def plot_feature_selection_progress(history, target, metric, out_path):
    """Plot the feature selection progress"""
    plt.figure(figsize=(12, 5))

    # Progress plot
    plt.subplot(1, 2, 1)
    plt.plot(history["k_features"], history[f"cv_{metric}"], 'o-', linewidth=2, markersize=6)
    plt.xlabel("Number of Features")
    plt.ylabel(f"CV {metric.replace('_', ' ').title()}")
    plt.title(f"{target.title()}: Feature Selection Progress")
    plt.grid(True, alpha=0.3)

    # Mark the best point
    best_idx = history[f"cv_{metric}"].idxmax()
    best_score = history[f"cv_{metric}"].iloc[best_idx]
    best_k = history["k_features"].iloc[best_idx]
    plt.scatter(best_k, best_score, color='red', s=100, zorder=5,
                label=f'Best: {best_k} features (score={best_score:.4f})')
    plt.legend()

    # Feature addition order (top 15)
    plt.subplot(1, 2, 2)
    top_features = history.head(15)
    y_pos = np.arange(len(top_features))
    scores = top_features[f"cv_{metric}"].values

    bars = plt.barh(y_pos, scores, alpha=0.7)
    plt.yticks(y_pos, [f"{i + 1}. {feat[:25]}{'...' if len(feat) > 25 else ''}"
                       for i, feat in enumerate(top_features["last_added"])])
    plt.xlabel(f"CV {metric.replace('_', ' ').title()}")
    plt.title(f"{target.title()}: Top 15 Features Added")
    plt.grid(True, alpha=0.3, axis='x')

    # Color the best bar differently
    if len(bars) > best_idx and best_idx < 15:
        bars[best_idx].set_color('red')
        bars[best_idx].set_alpha(0.9)

    plt.gca().invert_yaxis()  # First feature at top
    plt.tight_layout()
    plt.savefig(f"{out_path}/{target}_feature_selection_progress.png", dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_final_model_with_plots(model, Xte, yte, features, target, metric, out_path):
    """Comprehensive evaluation with plots similar to your other evaluation functions"""

    # Get predictions and probabilities
    y_prob = model.predict_proba(Xte[features])[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Calculate metrics
    acc = accuracy_score(yte, y_pred)
    prec = precision_score(yte, y_pred, zero_division=0)
    rec = recall_score(yte, y_pred, zero_division=0)
    f1 = f1_score(yte, y_pred, zero_division=0)
    cm = confusion_matrix(yte, y_pred, labels=[0, 1])
    ap = average_precision_score(yte, y_prob)
    auc = roc_auc_score(yte, y_prob)

    # Print results
    print(f"\n=== Final Model Evaluation: {target.title()} ===")
    print(f"Selected Features: {len(features)}")
    print(f"Prevalence (positives): {np.mean(yte):.3f}")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1:        {f1:.3f}")
    print(f"AUC-ROC:   {auc:.3f}")
    print(f"AUC-PR:    {ap:.3f}")

    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Confusion Matrix
    ax_cm = axes[0]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(values_format='d', cmap='Blues', ax=ax_cm, colorbar=False)
    ax_cm.set_title(f"{target.title()}: Confusion Matrix")

    # PR Curve
    ax_pr = axes[1]
    precision_curve, recall_curve, _ = precision_recall_curve(yte, y_prob)
    ax_pr.plot(recall_curve, precision_curve, lw=2)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"{target.title()}: PR Curve (AP = {ap:.3f})")
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.grid(True, alpha=0.3)

    # ROC Curve
    ax_roc = axes[2]
    fpr, tpr, _ = roc_curve(yte, y_prob)
    ax_roc.plot(fpr, tpr, lw=2)
    ax_roc.plot([0, 1], [0, 1], linestyle="--", lw=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"{target.title()}: ROC Curve (AUC = {auc:.3f})")
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{out_path}/{target}_final_model_evaluation.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Feature importance plot
    if hasattr(model, 'feature_importances_'):
        plot_feature_importance(model, features, target, out_path)

    # Return metrics dictionary
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "average_precision": float(ap),
        "confusion_matrix": cm.tolist(),
        "n_features": len(features),
        "prevalence": float(np.mean(yte))
    }


def plot_feature_importance(model, features, target, out_path, top_k=20):
    """Plot feature importance from the final model"""
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, max(6, top_k * 0.3)))
    top_features = feature_importance_df.head(top_k)

    bars = plt.barh(range(len(top_features)), top_features['importance'], alpha=0.8)
    plt.yticks(range(len(top_features)),
               [f"{feat[:40]}{'...' if len(feat) > 40 else ''}" for feat in top_features['feature']])
    plt.xlabel('Feature Importance (Gain)')
    plt.title(f'{target.title()}: Top {top_k} Feature Importances')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')

    # Color bars in gradient
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    plt.tight_layout()
    plt.savefig(f"{out_path}/{target}_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.show()

    return feature_importance_df


def train_final_xgb(Xtr, ytr, Xva, yva, Xte, yte, features, metric="average_precision", random_state=42):
    eval_metric = ("aucpr" if metric == "average_precision" else ("auc" if metric == "roc_auc" else "logloss"))
    pos_weight = len(ytr[ytr == 0]) / len(ytr[ytr == 1]) if ytr.sum() > 0 else 1

    params = dict(
        n_estimators=1000, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
        eval_metric=eval_metric, reg_lambda=1.0, scale_pos_weight=pos_weight
    )
    model = XGBClassifier(**params, early_stopping_rounds=50, random_state=random_state)
    model.fit(
        Xtr[features], ytr,
        eval_set=[(Xva[features], yva)],
        verbose=False
    )

    return model


def run_fs_and_eval_for_target(splits, target, max_features=60, patience=8, candidate_features=None,
                               groups_col="subject_id", out_path="./results"):
    # Create output directory
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # Unpack your splits
    Xtr, Xva, Xte = splits["X_train"], splits["X_val"], splits["X_test"]
    ytr, yva, yte = splits["y_train"][target].values, splits["y_val"][target].values, splits["y_test"][target].values
    feature_cols = splits["feature_cols"]

    # Optional subject-level grouping for CV (prevents leakage if multiple admissions per subject)
    groups = None
    if splits.get("ids_train") is not None and groups_col in splits["ids_train"].columns:
        groups = splits["ids_train"][groups_col].values

    # choose metric per task
    metric = "roc_auc" if target.lower() == "prolonged_stay" else "average_precision"

    print(f"\n=== {target}: feature selection on TRAIN (metric={metric}) ===")
    sel_feats, history = greedy_forward_select_xgb(
        Xtr[feature_cols], ytr, metric=metric,
        candidate_features=candidate_features,
        max_features=max_features, patience=patience,
        groups=groups
    )
    print(f"Selected {len(sel_feats)} features (first 10): {sel_feats[:10]}")

    # Plot feature selection progress
    plot_feature_selection_progress(history, target, metric, out_path)

    print(f"\n=== {target}: train on TRAIN (early-stop on VAL) & evaluate on TEST ===")
    model = train_final_xgb(
        Xtr, ytr, Xva, yva, Xte, yte, features=sel_feats, metric=metric
    )

    # Comprehensive evaluation with plots
    test_metrics = evaluate_final_model_with_plots(model, Xte, yte, sel_feats, target, metric, out_path)

    print("Final test metrics:", {k: v for k, v in test_metrics.items() if k != "confusion_matrix"})

    return {
        "selected_features": sel_feats,
        "history": history,
        "model": model,
        "test_metrics": test_metrics
    }


def save_results(res, target: str, out_path):
    out_path = Path(out_path)

    with open(f"{out_path}/{target}_selected_features.json", "w") as f:
        json.dump(res["selected_features"], f, indent=2)

    res["history"].to_csv(f"{out_path}/{target}_history.csv", index=False)

    with open(f"{out_path}/{target}_test_metrics.json", "w") as f:
        json.dump(res["test_metrics"], f, indent=2)

    try:
        res["model"].save_model(f"{out_path}/{target}_model.xgb")
    except Exception:
        dump(res["model"], f"{out_path}/{target}_model.pkl")

    print(f"✅ Saved all results for {target} in {out_path}")
    print(f"   📊 Plots: feature_selection_progress, final_model_evaluation, feature_importance")
    print(f"   📁 Files: selected_features.json, history.csv, test_metrics.json, model files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_pkl", required=True, help="Path to pickled splits (dumped from notebook)")
    ap.add_argument("--target", required=True, choices=["mortality", "prolonged_stay", "readmission"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_features", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    args = ap.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    splits = load(args.splits_pkl)
    res = run_fs_and_eval_for_target(
        splits,
        target=args.target,
        max_features=args.max_features,
        patience=args.patience,
        out_path=out_path
    )
    save_results(res, target=args.target, out_path=out_path)


if __name__ == '__main__':
    main()