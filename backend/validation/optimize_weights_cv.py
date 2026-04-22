import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

MATRIX = Path('backend/validation/artifacts/training_matrix.csv')
OUT_JSON = Path('backend/validation/artifacts/weight_optimization_results.json')
OUT_CSV = Path('backend/validation/artifacts/training_matrix_optimized_scores.csv')

FEATURES = ['expression_score', 'presentation_score', 'ccf', 'self_dissimilarity']
RL_SCORE = 'rl_priority'
BASELINE = 'binding_affinity_baseline_score'
LABEL = 'immunogenic'


def weighted_score(frame: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    x = frame[FEATURES].copy()
    x = x.fillna(0.0)
    return x.to_numpy() @ weights


def auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return float(roc_auc_score(y_true, y_score))


def simplex_grid(step: float = 0.05):
    k = int(round(1.0 / step))
    for a in range(k + 1):
        for b in range(k + 1 - a):
            for c in range(k + 1 - a - b):
                d = k - a - b - c
                yield np.array([a, b, c, d], dtype=float) * step


def main() -> int:
    tm = pd.read_csv(MATRIX)
    valid = tm[tm[LABEL].isin([0, 1])].copy()

    y = valid[LABEL].to_numpy(dtype=int)

    rl_auc = auc_safe(y, valid[RL_SCORE].fillna(0.0).to_numpy(dtype=float))
    baseline_auc = auc_safe(y, valid[BASELINE].fillna(0.0).to_numpy(dtype=float))

    print('=' * 70)
    print('WEIGHT OPTIMIZATION (5-fold CV)')
    print('=' * 70)
    print(f'Rows: {len(valid)} | Pos: {(y==1).sum()} | Neg: {(y==0).sum()}')
    print(f'Current RL AUC (in-sample): {rl_auc:.4f}')
    print(f'Baseline AUC (in-sample):   {baseline_auc:.4f}')

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_w = None
    best_auc = -1.0
    tested = 0

    for w in simplex_grid(step=0.05):
        fold_aucs = []
        for train_idx, test_idx in cv.split(valid, y):
            test_df = valid.iloc[test_idx]
            y_test = y[test_idx]
            s_test = weighted_score(test_df, w)
            fold_aucs.append(auc_safe(y_test, s_test))
        m = float(np.nanmean(fold_aucs))
        tested += 1
        if m > best_auc:
            best_auc = m
            best_w = w.copy()

    print(f'Grid searched: {tested} weight vectors')
    print(f'Best CV AUC: {best_auc:.4f}')
    print('Best weights:')
    for name, val in zip(FEATURES, best_w, strict=True):
        print(f'  {name}: {val:.2f}')

    # Fold-level report for best weights
    fold_details = []
    for i, (_, test_idx) in enumerate(cv.split(valid, y), start=1):
        test_df = valid.iloc[test_idx]
        y_test = y[test_idx]
        s_test = weighted_score(test_df, best_w)
        auc = auc_safe(y_test, s_test)
        fold_details.append(float(auc))
        print(f'  Fold {i}: AUC={auc:.4f} (n={len(test_idx)}, pos={(y_test==1).sum()})')

    # In-sample with optimized weights (for reference only)
    valid['optimized_priority'] = weighted_score(valid, best_w)
    opt_insample_auc = auc_safe(y, valid['optimized_priority'].to_numpy(dtype=float))
    print(f'Optimized in-sample AUC (reference): {opt_insample_auc:.4f}')
    print(f'CV gain vs current RL: {best_auc - rl_auc:+.4f}')
    print(f'CV gain vs baseline:   {best_auc - baseline_auc:+.4f}')

    # Save artifacts
    out = {
        'rows': int(len(valid)),
        'positives': int((y == 1).sum()),
        'negatives': int((y == 0).sum()),
        'features': FEATURES,
        'current_rl_auc_insample': rl_auc,
        'baseline_auc_insample': baseline_auc,
        'cv_best_auc': best_auc,
        'cv_fold_aucs': fold_details,
        'best_weights': {k: float(v) for k, v in zip(FEATURES, best_w, strict=True)},
        'optimized_auc_insample': opt_insample_auc,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding='utf-8')

    merged = tm.copy()
    merged['optimized_priority'] = weighted_score(merged, best_w)
    merged.to_csv(OUT_CSV, index=False)
    print(f'Saved: {OUT_JSON}')
    print(f'Saved: {OUT_CSV}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
