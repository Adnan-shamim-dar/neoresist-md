from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'backend' / 'validation' / 'artifacts'


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors='coerce')


def _auc(y: pd.Series, s: pd.Series) -> float:
    yv = _num(y)
    sv = _num(s)
    m = yv.isin([0, 1]) & sv.notna()
    if m.sum() < 10 or yv[m].nunique() < 2:
        return float('nan')
    return float(roc_auc_score(yv[m].astype(int), sv[m].astype(float)))


def _bootstrap_auc_delta(y: pd.Series, s1: pd.Series, s2: pd.Series, n: int = 1000, seed: int = 42) -> dict[str, float]:
    y = _num(y)
    s1 = _num(s1)
    s2 = _num(s2)
    m = y.isin([0, 1]) & s1.notna() & s2.notna()
    y = y[m].astype(int).to_numpy()
    s1 = s1[m].astype(float).to_numpy()
    s2 = s2[m].astype(float).to_numpy()
    if len(y) < 20 or len(np.unique(y)) < 2:
        return {'delta': float('nan'), 'ci_low': float('nan'), 'ci_high': float('nan'), 'p_bootstrap': float('nan')}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        idx = rng.integers(0, len(y), size=len(y))
        ys = y[idx]
        if len(np.unique(ys)) < 2:
            continue
        d = roc_auc_score(ys, s1[idx]) - roc_auc_score(ys, s2[idx])
        deltas.append(float(d))
    if not deltas:
        return {'delta': float('nan'), 'ci_low': float('nan'), 'ci_high': float('nan'), 'p_bootstrap': float('nan')}
    arr = np.array(deltas)
    delta = float(roc_auc_score(y, s1) - roc_auc_score(y, s2))
    p = float(2.0 * min((arr <= 0).mean(), (arr >= 0).mean()))
    return {
        'delta': delta,
        'ci_low': float(np.quantile(arr, 0.025)),
        'ci_high': float(np.quantile(arr, 0.975)),
        'p_bootstrap': p,
    }


def _dynamic_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    feats = list(weights)

    def score_row(r: pd.Series) -> float:
        avail = []
        for f in feats:
            v = r.get(f)
            if pd.notna(v) and float(v) > 0:
                avail.append((float(v), float(weights[f])))
        if not avail:
            return float('nan')
        tw = sum(w for _, w in avail)
        if tw <= 0:
            return float('nan')
        return float(sum(v * (w / tw) for v, w in avail))

    return df.apply(score_row, axis=1)


def _collapse_mutation_level(ott: pd.DataFrame) -> pd.DataFrame:
    key = ['patient_id', 'gene', 'protein_change']
    g = ott.groupby(key, dropna=False)

    def pick_hla(s: pd.Series) -> str:
        idx = s.fillna(-1).astype(float).idxmax()
        return str(ott.loc[idx, 'hla_allele']) if pd.notna(idx) else ''

    out = g.agg(
        immunogenic=('immunogenic', 'max'),
        expression_score=('expression_score', 'first'),
        self_dissimilarity=('self_dissimilarity', 'mean'),
        presentation_score=('presentation_score', 'max'),
        binding_affinity_baseline_score=('binding_affinity_baseline_score', 'max'),
        paper_source=('paper_source', 'first'),
        cancer_type=('cancer_type', 'first'),
        peptide_rows=('mutant_peptide', 'count'),
    ).reset_index()
    out['best_hla_allele'] = g['presentation_score'].apply(pick_hla).values
    return out


def _grid_optimize(df: pd.DataFrame, label_col: str = 'immunogenic', step: float = 0.05) -> tuple[dict[str, float], float]:
    feats = ['expression_score', 'presentation_score', 'self_dissimilarity']
    vals = [round(i * step, 10) for i in range(int(1 / step) + 1)]
    y = _num(df[label_col])
    best_auc = -1.0
    best_w = {'expression_score': 0.2, 'presentation_score': 0.3, 'self_dissimilarity': 0.1}
    for w1, w2, w3 in product(vals, vals, vals):
        if w1 + w2 + w3 == 0:
            continue
        w = {'expression_score': w1, 'presentation_score': w2, 'self_dissimilarity': w3}
        s = _dynamic_score(df, w)
        auc = _auc(y, s)
        if np.isfinite(auc) and auc > best_auc:
            best_auc = auc
            best_w = w
    return best_w, float(best_auc)


def _lopo_cv(df: pd.DataFrame) -> dict[str, object]:
    patients = sorted(df['patient_id'].dropna().astype(str).unique().tolist())
    out: dict[str, float] = {}
    for p in patients:
        tr = df[df['patient_id'] != p]
        te = df[df['patient_id'] == p]
        if tr['immunogenic'].nunique() < 2 or te['immunogenic'].nunique() < 2:
            out[p] = float('nan')
            continue
        w, _ = _grid_optimize(tr, step=0.1)
        s = _dynamic_score(te, w)
        out[p] = _auc(te['immunogenic'], s)
    vals = [v for v in out.values() if np.isfinite(v)]
    return {
        'per_patient_auc': out,
        'mean_auc': float(np.mean(vals)) if vals else float('nan'),
        'median_auc': float(np.median(vals)) if vals else float('nan'),
    }


def _patient_rankings(df: pd.DataFrame, score_cols: dict[str, str]) -> pd.DataFrame:
    rows = []
    for p, sub in df.groupby('patient_id'):
        imm = sub[sub['immunogenic'] == 1]
        for label, col in score_cols.items():
            rank = sub[col].rank(ascending=False, method='min')
            r_imm = rank.loc[imm.index]
            rows.append(
                {
                    'patient_id': p,
                    'score': label,
                    'haystack_n': int(len(sub)),
                    'immunogenic_n': int(len(imm)),
                    'median_rank_immunogenic': float(r_imm.median()) if len(r_imm) else float('nan'),
                    'median_percentile_immunogenic': float((r_imm / len(sub)).median()) if len(r_imm) else float('nan'),
                    'recall_at_20': float((r_imm <= 20).sum() / max(len(imm), 1)),
                    'recall_at_50': float((r_imm <= 50).sum() / max(len(imm), 1)),
                    'auc': _auc(sub['immunogenic'], sub[col]),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)

    tm = pd.read_csv(ART / 'training_matrix.csv')
    ott = tm[tm['paper_source'] == 'ott_2017'].copy()
    for c in ['immunogenic', 'expression_score', 'presentation_score', 'self_dissimilarity', 'binding_affinity_baseline_score']:
        if c in ott.columns:
            ott[c] = _num(ott[c])

    print('=' * 70)
    print('OTT 2017 HAYSTACK ANALYSIS — DIAGNOSTICS')
    print('=' * 70)

    # CHECK 1
    print('\nCHECK 1: Ott data availability')
    print(f"  Ott peptide rows: {len(ott)}")
    print(f"  Immunogenic: {int(ott['immunogenic'].sum())} / Non-immunogenic: {int((ott['immunogenic']==0).sum())}")
    print(f"  Patients: {ott['patient_id'].nunique()}")
    for p, g in ott.groupby('patient_id'):
        print(f"    {p}: n={len(g)}, immunogenic={int(g['immunogenic'].sum())}")
    print(f"  expression > 0: {int((ott['expression_score']>0).sum())}")
    print(f"  presentation > 0: {int((ott['presentation_score']>0).sum())}")
    sd_ok = ott['self_dissimilarity'].notna() & (ott['self_dissimilarity'] != 0.5) & (ott['self_dissimilarity'] != 0)
    print(f"  self_dissimilarity valid: {int(sd_ok.sum())}")

    # CHECK 2
    hay = pd.read_csv(ROOT / 'validation_papers' / 'merged_somatic_mutations.csv')
    ott_hay = hay[hay['paper_source'].astype(str).str.lower().eq('ott_2017')].copy()
    expr_cols = [
        c
        for c in ott_hay.columns
        if ('tpm' in c.lower())
        or ('expression' in c.lower())
        or (c.lower() in {'rna', 'rna_tpm', 'expression_tpm'})
    ]
    pep_cols = [c for c in ott_hay.columns if 'peptide' in c.lower()]
    print('\nCHECK 2: Haystack availability')
    print(f"  Ott haystack rows: {len(ott_hay)}")
    print(f"  Columns: {list(ott_hay.columns)}")
    print(f"  Expression-like columns: {expr_cols}")
    print(f"  Peptide columns: {pep_cols}")
    print(ott_hay.head(5).to_string(index=False))

    # CHECK 3
    key = ['patient_id', 'gene', 'protein_change']
    n_unique = ott[key].drop_duplicates().shape[0]
    inflation = len(ott) / max(n_unique, 1)
    print('\nCHECK 3: Sub-peptide inflation')
    print(f"  peptide rows={len(ott)}, unique mutations={n_unique}, inflation_ratio={inflation:.3f}")
    print(ott.groupby(key).size().value_counts().sort_index().to_string())

    # CHECK 4/5
    print('\nCHECK 4: expression transform sample')
    print(ott[['expression_tpm', 'expression_tpm_numeric', 'expression_score']].dropna().head(20).to_string(index=False))
    print('\nCHECK 5: binding transform verification sample')
    s = ott[['binding_affinity_nm_numeric', 'presentation_score']].dropna().head(20).copy()
    s['expected'] = (1 - (np.log10(s['binding_affinity_nm_numeric'].clip(lower=1e-9)) / np.log10(50000))).clip(0, 1)
    s['delta'] = s['presentation_score'] - s['expected']
    print(s.to_string(index=False))
    print(f"  max abs delta={float(s['delta'].abs().max()):.6f}")

    # CHECK 6
    hla = pd.read_csv(ROOT / 'validation_papers' / 'patient_hla_alleles.csv')
    print('\nCHECK 6: HLA alleles')
    print(hla[hla['patient_id'].astype(str).str.startswith('ott_')].to_string(index=False))

    missing_haystack_features = (len(expr_cols) == 0) or (len(pep_cols) == 0)
    if missing_haystack_features:
        print('\nCHECK STATUS: FAIL for full mutanome scoring (missing expression and/or peptide sequences).')
        print('Switching to fallback mode on labeled Ott set only, per protocol.')

    # Build mutation-level labeled
    ott_mut = _collapse_mutation_level(ott)

    # Scoring (dynamic renorm, CCF excluded)
    start_w = {'expression_score': 0.20, 'presentation_score': 0.30, 'self_dissimilarity': 0.10}
    ott['dynamic_rl_score'] = _dynamic_score(ott, start_w)
    ott_mut['dynamic_rl_score'] = _dynamic_score(ott_mut, start_w)

    # fallback haystack scored: map known labeled mutation scores, leave unknown NaN
    key_cols = ['patient_id', 'gene', 'protein_change']
    known = ott_mut[key_cols + ['dynamic_rl_score', 'presentation_score', 'expression_score', 'self_dissimilarity', 'immunogenic']].copy()
    known = known.rename(columns={'immunogenic': 'immunogenic_labeled'})
    ott_hay_scored = ott_hay.merge(known, on=key_cols, how='left')
    ott_hay_scored['is_labeled_mutation'] = ott_hay_scored['immunogenic_labeled'].notna().astype(int)

    # Per-patient rankings on labeled candidate sets
    rank_pep = _patient_rankings(ott, {'binding_only_score': 'presentation_score', 'dynamic_rl_score': 'dynamic_rl_score'})
    rank_mut = _patient_rankings(ott_mut, {'binding_only_score': 'presentation_score', 'dynamic_rl_score': 'dynamic_rl_score'})
    rankings = pd.concat([rank_pep.assign(level='peptide'), rank_mut.assign(level='mutation')], ignore_index=True)

    # primary AUC on mutation-level labeled (honest fallback)
    auc_bind = _auc(ott_mut['immunogenic'], ott_mut['presentation_score'])
    auc_dyn = _auc(ott_mut['immunogenic'], ott_mut['dynamic_rl_score'])
    bs = _bootstrap_auc_delta(ott_mut['immunogenic'], ott_mut['dynamic_rl_score'], ott_mut['presentation_score'])

    # ablation on mutation level
    combos = {
        'all_three': ['expression_score', 'presentation_score', 'self_dissimilarity'],
        'drop_expression': ['presentation_score', 'self_dissimilarity'],
        'drop_presentation': ['expression_score', 'self_dissimilarity'],
        'drop_dissimilarity': ['expression_score', 'presentation_score'],
        'presentation_only': ['presentation_score'],
        'expression_only': ['expression_score'],
        'dissimilarity_only': ['self_dissimilarity'],
    }
    ablation = {}
    for name, cols in combos.items():
        w = {c: 1.0 for c in cols}
        s = _dynamic_score(ott_mut, w)
        ablation[name] = _auc(ott_mut['immunogenic'], s)

    # optimize weights on mutation-level
    best_w, best_auc = _grid_optimize(ott_mut, step=0.05)

    # top 20 weights
    tops = []
    vals = [round(i * 0.05, 10) for i in range(21)]
    for w1, w2, w3 in product(vals, vals, vals):
        if w1 + w2 + w3 == 0:
            continue
        w = {'expression_score': w1, 'presentation_score': w2, 'self_dissimilarity': w3}
        s = _dynamic_score(ott_mut, w)
        a = _auc(ott_mut['immunogenic'], s)
        if np.isfinite(a):
            tops.append({'auc': float(a), **w})
    top20 = sorted(tops, key=lambda x: x['auc'], reverse=True)[:20]

    # logistic regression (labeled mutation-level)
    feats = ['expression_score', 'presentation_score', 'self_dissimilarity']
    X = ott_mut[feats].fillna(0.0).to_numpy()
    y = ott_mut['immunogenic'].astype(int).to_numpy()
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(Xs, y)
    lr_prob = lr.predict_proba(Xs)[:, 1]
    lr_auc = float(roc_auc_score(y, lr_prob))

    # LOPO CV
    lopo = _lopo_cv(ott_mut)

    # SHANK2 case
    shank = tm[(tm['patient_id'].astype(str) == 'keskin_8') & (tm['gene'].astype(str).str.upper() == 'SHANK2')].copy()
    shank_info = {}
    if not shank.empty:
        row = shank.iloc[0]
        keskin = tm[tm['patient_id'].astype(str) == 'keskin_8'].copy()
        for c in ['presentation_score', 'rl_priority', 'expression_score', 'self_dissimilarity', 'binding_affinity_nm_numeric']:
            if c in keskin.columns:
                keskin[c] = _num(keskin[c])
        rank_bind = int((keskin['presentation_score'].rank(ascending=False, method='min').loc[row.name])) if 'presentation_score' in keskin.columns else None
        rank_rl = int((keskin['rl_priority'].rank(ascending=False, method='min').loc[row.name])) if 'rl_priority' in keskin.columns else None
        shank_info = {
            'patient_id': 'keskin_8',
            'gene': 'SHANK2',
            'binding_affinity_nm': float(row.get('binding_affinity_nm_numeric')) if pd.notna(row.get('binding_affinity_nm_numeric')) else None,
            'presentation_score': float(row.get('presentation_score')) if pd.notna(row.get('presentation_score')) else None,
            'expression_score': float(row.get('expression_score')) if pd.notna(row.get('expression_score')) else None,
            'self_dissimilarity': float(row.get('self_dissimilarity')) if pd.notna(row.get('self_dissimilarity')) else None,
            'rl_priority': float(row.get('rl_priority')) if pd.notna(row.get('rl_priority')) else None,
            'binding_rank_within_keskin8': rank_bind,
            'rl_rank_within_keskin8': rank_rl,
            'n_keskin8_rows': int(len(keskin)),
        }

    # Save artifacts
    ott_hay_scored.to_csv(ART / 'ott_haystack_scored.csv', index=False)
    ott_mut.to_csv(ART / 'ott_labeled_mutation_level.csv', index=False)
    rankings.to_csv(ART / 'ott_patient_rankings.csv', index=False)

    results = {
        'mode': 'fallback_labeled_only' if missing_haystack_features else 'full_haystack',
        'limitations': {
            'missing_haystack_expression': len(expr_cols) == 0,
            'missing_haystack_peptides': len(pep_cols) == 0,
            'note': 'Unlabeled haystack rows are not true negatives; all AUCs here are lower-bound or fallback estimates.'
        },
        'dataset': {
            'patients': int(ott['patient_id'].nunique()),
            'vaccine_peptides': int(len(ott)),
            'immunogenic_peptides': int(ott['immunogenic'].sum()),
            'mutation_level_rows': int(len(ott_mut)),
            'mutation_level_immunogenic': int(ott_mut['immunogenic'].sum()),
            'haystack_rows': int(len(ott_hay_scored)),
            'haystack_labeled_overlap': int(ott_hay_scored['is_labeled_mutation'].sum()),
        },
        'primary_result': {
            'binding_only_auc': auc_bind,
            'dynamic_rl_auc': auc_dyn,
            'delta': float(auc_dyn - auc_bind) if np.isfinite(auc_dyn) and np.isfinite(auc_bind) else float('nan'),
            'auc_delta_ci95': [bs['ci_low'], bs['ci_high']],
            'p_value_bootstrap': bs['p_bootstrap'],
            'n_haystack': int(len(ott_hay_scored)),
            'n_immunogenic': int(ott_mut['immunogenic'].sum()),
            'n_patients': int(ott['patient_id'].nunique()),
        },
        'per_patient': rankings.to_dict(orient='records'),
        'ablation': ablation,
        'optimal_weights': {
            'grid_best': best_w,
            'grid_best_auc': best_auc,
            'top20': top20,
            'logistic_regression_auc': lr_auc,
            'logistic_coefficients': {f: float(c) for f, c in zip(feats, lr.coef_[0], strict=True)},
        },
        'lopo_cv': lopo,
        'shank2_case': shank_info,
    }
    (ART / 'ott_haystack_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')

    # terminal summary
    print('\n' + '=' * 70)
    print('OTT 2017 HAYSTACK ANALYSIS — RESULTS')
    print('=' * 70)
    print('DATASET:')
    print(f"  Patients: {results['dataset']['patients']}")
    print(f"  Total mutations (haystack): {results['dataset']['haystack_rows']}")
    print(f"  Vaccine-selected peptides: {results['dataset']['vaccine_peptides']}")
    print(f"  Immunogenic peptides: {results['dataset']['immunogenic_peptides']}")
    print(f"  Mutation-level: {results['dataset']['mutation_level_rows']} unique, {results['dataset']['mutation_level_immunogenic']} immunogenic")

    print('\nPRIMARY RESULT:')
    print(f"  Binding-only AUC:    {auc_bind:.4f}")
    print(f"  Dynamic RL AUC:      {auc_dyn:.4f}")
    print(f"  Delta:              {auc_dyn-auc_bind:+.4f}")
    print(f"  p-value (bootstrap): {bs['p_bootstrap']:.4f}")
    print(f"  95% CI delta:        [{bs['ci_low']:.4f}, {bs['ci_high']:.4f}]")

    print('\nPER-PATIENT:')
    show = rankings[['level', 'patient_id', 'score', 'haystack_n', 'immunogenic_n', 'median_rank_immunogenic', 'recall_at_20', 'recall_at_50', 'auc']]
    print(show.to_string(index=False))

    print('\nABLATION:')
    for k, v in ablation.items():
        print(f"  {k:<20} {v:.4f}")

    print('\nOPTIMAL WEIGHTS (grid):')
    print(f"  expression: {best_w['expression_score']:.2f}")
    print(f"  presentation: {best_w['presentation_score']:.2f}")
    print(f"  self_dissimilarity: {best_w['self_dissimilarity']:.2f}")
    print(f"  Optimal AUC: {best_auc:.4f}")

    print('\nCROSS-VALIDATION (LOPO):')
    print(f"  Mean LOPO AUC: {results['lopo_cv']['mean_auc']:.4f}")
    for p, a in results['lopo_cv']['per_patient_auc'].items():
        print(f"  {p}: {a:.4f}")

    print('\nSHANK2 CASE STUDY:')
    if shank_info:
        print(json.dumps(shank_info, indent=2))
    else:
        print('  SHANK2 row not found in training_matrix.csv')

    print('\nLIMITATION MODE:')
    print(f"  mode={results['mode']}")
    print(f"  missing_haystack_expression={results['limitations']['missing_haystack_expression']}")
    print(f"  missing_haystack_peptides={results['limitations']['missing_haystack_peptides']}")
    print('=' * 70)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
