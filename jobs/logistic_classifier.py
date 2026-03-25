"""logistic regression + ROC for predicting self-replication from derived measures."""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run(artifact: Path, boundary_path: Path = None, oinfo_path: Path = None, **kwargs) -> dict:
    """fit logistic regression on (lambda, F, mass-balance, O-info) -> tier1 label."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    if boundary_path is None:
        boundary_path = Path("results/k2-moore-boundary-measures.json")
    if oinfo_path is None:
        oinfo_path = Path("results/k2-moore-oinfo-boundary.json")

    bm_data = json.loads(boundary_path.read_text())
    oi_data = json.loads(oinfo_path.read_text())

    bm_by_idx = {r["rule_index"]: r for r in bm_data["rules"]}
    oi_by_idx = {r["rule_index"]: r for r in oi_data["rules"]}

    common_idx = set(bm_by_idx.keys()) & set(oi_by_idx.keys())
    feature_names = ["lambda", "f_param", "mass_balance", "spatial_entropy", "oinfo"]

    X_rows, y_rows = [], []
    for idx in sorted(common_idx):
        bm_r, oi_r = bm_by_idx[idx], oi_by_idx[idx]
        row = [
            bm_r["lambda"], bm_r.get("f_param", bm_r.get("f", 0)),
            bm_r["mass_balance"], bm_r["spatial_entropy"],
            oi_r["oinfo"],
        ]
        X_rows.append(row)
        y_rows.append(1 if bm_r["label"] == "tier1_positive" else 0)

    X = np.array(X_rows)
    y = np.array(y_rows)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, random_state=42)
    probs = cross_val_predict(model, X_scaled, y, cv=5, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, probs)

    model.fit(X_scaled, y)
    coefs = dict(zip(feature_names, [round(float(c), 4) for c in model.coef_[0]]))

    fpr, tpr = _roc_curve(y, probs)

    results = {
        "n_samples": len(y),
        "n_positive": int(y.sum()),
        "auc_5fold_cv": round(float(auc), 4),
        "coefficients": coefs,
        "intercept": round(float(model.intercept_[0]), 4),
        "feature_names": feature_names,
        "roc_fpr": fpr,
        "roc_tpr": tpr,
    }

    artifact.write_text(json.dumps(results, indent=2, default=str))
    logger.info("  logistic classifier: AUC=%.3f, coefs=%s", auc, coefs)
    return results


def _roc_curve(y_true, y_prob, n_points: int = 50):
    """compute ROC curve at n_points thresholds."""
    import numpy as np
    thresholds = np.linspace(0, 1, n_points)
    fpr_list, tpr_list = [], []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        tn = ((pred == 0) & (y_true == 0)).sum()
        fpr_list.append(round(float(fp / max(fp + tn, 1)), 4))
        tpr_list.append(round(float(tp / max(tp + fn, 1)), 4))
    return fpr_list, tpr_list
