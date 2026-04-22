import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

tm = pd.read_csv('backend/validation/artifacts/training_matrix.csv')

# Use correct column names
FEATURES = {
    'expression_score': 'expression_score',
    'presentation_score': 'presentation_score',
    'ccf': 'ccf',
    'self_dissimilarity': 'self_dissimilarity',
}
RL_SCORE = 'rl_priority'
BASELINE = 'binding_affinity_baseline_score'

# All 701 rows are usable — no filter needed
valid = tm[tm['immunogenic'].isin([0, 1])].copy()
immuno = valid[valid['immunogenic'] == 1]
non_immuno = valid[valid['immunogenic'] == 0]

print("=" * 70)
print("NEORESIST-MD EXPLORATORY DATA ANALYSIS")
print("=" * 70)
print(f"Total rows: {len(valid)}")
print(f"Immunogenic (1): {len(immuno)}")
print(f"Non-immunogenic (0): {len(non_immuno)}")
print(f"Positive rate: {100*len(immuno)/len(valid):.1f}%")

# ===================================================
print(f"\n{'='*70}")
print("1A: SINGLE-FEATURE AUC-ROC")
print("Does each feature ALONE predict immunogenicity?")
print(f"{'='*70}")

single_aucs = {}
for display_name, col in FEATURES.items():
    subset = valid.dropna(subset=[col])
    # Also remove zeros if they might be NaN-substitutes
    subset_nonzero = subset[subset[col] != 0]
    
    for label, data in [("all", subset), 
                         ("nonzero only", subset_nonzero)]:
        if len(data) > 20 and data['immunogenic'].nunique() == 2:
            auc = roc_auc_score(data['immunogenic'], data[col])
            i_vals = data[data['immunogenic']==1][col]
            n_vals = data[data['immunogenic']==0][col]
            
            try:
                stat, pval = stats.mannwhitneyu(
                    i_vals, n_vals, alternative='two-sided')
            except:
                pval = 1.0
            
            direction = "higher" if i_vals.mean() > n_vals.mean() \
                        else "LOWER"
            
            if label == "all":
                single_aucs[display_name] = auc
            
            print(f"\n{display_name} ({label}, n={len(data)}, "
                  f"pos={int(data['immunogenic'].sum())}):")
            print(f"  AUC = {auc:.4f}")
            print(f"  Immunogenic mean:     {i_vals.mean():.4f} "
                  f"(median={i_vals.median():.4f})")
            print(f"  Non-immunogenic mean: {n_vals.mean():.4f} "
                  f"(median={n_vals.median():.4f})")
            print(f"  Direction: immunogenic scores {direction}")
            print(f"  Mann-Whitney p = {pval:.4e}")
            if auc > 0.55 and pval < 0.05:
                print(f"  → INFORMATIVE ✅")
            elif auc > 0.52:
                print(f"  → WEAK SIGNAL ⚠️")
            else:
                print(f"  → NOT PREDICTIVE ❌")
        else:
            if label == "all":
                single_aucs[display_name] = None
            print(f"\n{display_name} ({label}): "
                  f"insufficient data (n={len(data)})")

# Composite and baseline
print(f"\n--- Composite scores ---")
for name, col in [(RL_SCORE, RL_SCORE), 
                   ("baseline", BASELINE)]:
    subset = valid.dropna(subset=[col])
    subset = subset[subset[col] > 0]
    if len(subset) > 20 and subset['immunogenic'].nunique() == 2:
        auc = roc_auc_score(subset['immunogenic'], subset[col])
        print(f"\n{name} (n={len(subset)}):")
        print(f"  AUC = {auc:.4f}")

# ===================================================
print(f"\n{'='*70}")
print("1B: PER-PAPER SINGLE-FEATURE AUC")
print(f"{'='*70}")

feat_cols = list(FEATURES.values())
papers = sorted(valid['paper_source'].unique())

header = f"{'Paper':<15}"
for f in FEATURES:
    header += f" {f[:12]:>12}"
header += f" {'rl_priority':>12} {'baseline':>12}"
print(header)
print("-" * len(header))

for paper in papers:
    sub = valid[valid['paper_source'] == paper]
    n_pos = int((sub['immunogenic'] == 1).sum())
    n_neg = int((sub['immunogenic'] == 0).sum())
    line = f"{paper:<15}"
    
    for feat_name, feat_col in FEATURES.items():
        feat_sub = sub.dropna(subset=[feat_col])
        if len(feat_sub) > 10 and \
           feat_sub['immunogenic'].nunique() == 2:
            auc = roc_auc_score(
                feat_sub['immunogenic'], feat_sub[feat_col])
            line += f" {auc:>12.4f}"
        else:
            line += f" {'N/A':>12}"
    
    # RL
    rl_sub = sub.dropna(subset=[RL_SCORE])
    if len(rl_sub) > 10 and rl_sub['immunogenic'].nunique() == 2:
        auc = roc_auc_score(
            rl_sub['immunogenic'], rl_sub[RL_SCORE])
        line += f" {auc:>12.4f}"
    else:
        line += f" {'N/A':>12}"
    
    # Baseline
    bl_sub = sub.dropna(subset=[BASELINE])
    bl_sub = bl_sub[bl_sub[BASELINE] > 0]
    if len(bl_sub) > 10 and bl_sub['immunogenic'].nunique() == 2:
        auc = roc_auc_score(
            bl_sub['immunogenic'], bl_sub[BASELINE])
        line += f" {auc:>12.4f}"
    else:
        line += f" {'N/A':>12}"
    
    line += f"  (+={n_pos}, -={n_neg})"
    print(line)

# ===================================================
print(f"\n{'='*70}")
print("1C: FEATURE DISTRIBUTIONS")
print(f"{'='*70}")

for feat_name, feat_col in FEATURES.items():
    vals = valid[feat_col].dropna()
    if len(vals) > 10:
        i_v = immuno[feat_col].dropna()
        n_v = non_immuno[feat_col].dropna()
        print(f"\n{feat_name} (col: {feat_col}):")
        print(f"  All:      n={len(vals):>4}  "
              f"min={vals.min():.4f}  "
              f"med={vals.median():.4f}  "
              f"mean={vals.mean():.4f}  "
              f"max={vals.max():.4f}  "
              f"std={vals.std():.4f}")
        print(f"  Zeros:    {(vals == 0).sum()}")
        if len(i_v) > 3:
            print(f"  Immuno:   n={len(i_v):>4}  "
                  f"min={i_v.min():.4f}  "
                  f"med={i_v.median():.4f}  "
                  f"mean={i_v.mean():.4f}  "
                  f"max={i_v.max():.4f}")
        if len(n_v) > 3:
            print(f"  Non-imm:  n={len(n_v):>4}  "
                  f"min={n_v.min():.4f}  "
                  f"med={n_v.median():.4f}  "
                  f"mean={n_v.mean():.4f}  "
                  f"max={n_v.max():.4f}")
    else:
        print(f"\n{feat_name}: only {len(vals)} values")

# ===================================================
print(f"\n{'='*70}")
print("1D: FEATURE CORRELATIONS")
print(f"{'='*70}")

# Pairwise (since not all rows have all features)
for i, (n1, c1) in enumerate(FEATURES.items()):
    for n2, c2 in list(FEATURES.items())[i+1:]:
        pair = valid[[c1, c2]].dropna()
        if len(pair) > 20:
            r = pair[c1].corr(pair[c2])
            print(f"  {n1} vs {n2}: r={r:.3f} (n={len(pair)})")

# Correlation with immunogenic
print(f"\nCorrelation with immunogenic:")
for feat_name, feat_col in FEATURES.items():
    both = valid[['immunogenic', feat_col]].dropna()
    if len(both) > 20:
        r = both[feat_col].corr(both['immunogenic'])
        print(f"  {feat_name}: r={r:.4f} (n={len(both)})")

# Correlation between components and rl_priority
print(f"\nCorrelation with rl_priority:")
for feat_name, feat_col in FEATURES.items():
    both = valid[[RL_SCORE, feat_col]].dropna()
    if len(both) > 20:
        r = both[feat_col].corr(both[RL_SCORE])
        print(f"  {feat_name}: r={r:.4f} (n={len(both)})")

# ===================================================
print(f"\n{'='*70}")
print("1E: COMPOSITE vs BEST SINGLE FEATURE")
print(f"{'='*70}")

valid_aucs = {k: v for k, v in single_aucs.items() 
              if v is not None}
if valid_aucs:
    best_feat, best_auc = max(valid_aucs.items(), 
                               key=lambda x: x[1])
    print(f"Best single feature: {best_feat} "
          f"(AUC={best_auc:.4f})")
    
    rl_sub = valid.dropna(subset=[RL_SCORE])
    if len(rl_sub) > 20:
        auc_comp = roc_auc_score(
            rl_sub['immunogenic'], rl_sub[RL_SCORE])
        print(f"RL composite AUC:   {auc_comp:.4f}")
        delta = auc_comp - best_auc
        print(f"Delta: {delta:+.4f}")
        if delta > 0.02:
            print("→ Composite BEATS best single feature ✅")
        elif delta > -0.02:
            print("→ Composite roughly MATCHES single feature ⚠️")
        else:
            print("→ Composite WORSE than best single feature ❌")
            print("  Current weights blend in noisy features")

# ===================================================
print(f"\n{'='*70}")
print("1F: SHANK2 CASE STUDY")
print(f"{'='*70}")

shank = valid[valid['gene'].astype(str).str.contains(
    'SHANK', case=False, na=False)]
if len(shank) > 0:
    for _, row in shank.iterrows():
        print(f"  Patient: {row['patient_id']}")
        print(f"  Gene: {row['gene']}")
        print(f"  Immunogenic: {int(row['immunogenic'])}")
        rl = row[RL_SCORE]
        bl = row[BASELINE]
        print(f"  RL: {rl:.4f}" if pd.notna(rl) else "  RL: NaN")
        print(f"  Baseline: {bl:.4f}" if pd.notna(bl) 
              else "  Baseline: NaN")
        for fn, fc in FEATURES.items():
            v = row[fc]
            print(f"  {fn}: {v:.4f}" if pd.notna(v) 
                  else f"  {fn}: NaN")
        
        # Rank within patient
        pat = valid[valid['patient_id'] == row['patient_id']]
        pat_sorted_rl = pat.sort_values(RL_SCORE, ascending=False)
        pat_sorted_bl = pat.sort_values(BASELINE, ascending=False)
        rank_rl = (pat_sorted_rl[RL_SCORE].values >= rl).sum()
        rank_bl = (pat_sorted_bl[BASELINE].values >= bl).sum()
        print(f"  RL rank: {rank_rl}/{len(pat)}")
        print(f"  Baseline rank: {rank_bl}/{len(pat)}")
else:
    print("  SHANK2 not found")
    keskin = valid[valid['paper_source'].str.contains(
        'keskin', case=False, na=False)]
    print(f"  Keskin rows: {len(keskin)}")
    if len(keskin) > 0:
        print(f"  Keskin genes: "
              f"{sorted(keskin['gene'].unique().tolist())}")
        imm_genes = keskin[keskin['immunogenic']==1][
            'gene'].unique().tolist()
        print(f"  Keskin immunogenic genes: {sorted(imm_genes)}")

# ===================================================
print(f"\n{'='*70}")
print("1G: FEATURE AVAILABILITY MATRIX")
print(f"{'='*70}")

# Count features available per row
feat_cols = list(FEATURES.values())
n_feat = valid[feat_cols].notna().sum(axis=1)
# Don't count zeros as available for ccf
n_feat_strict = pd.Series(0, index=valid.index)
for fc in feat_cols:
    has_val = valid[fc].notna() & (valid[fc] != 0)
    n_feat_strict += has_val.astype(int)

print(f"\nFeatures available per row (counting non-NaN):")
for n in range(5):
    count = (n_feat == n).sum()
    pct = 100 * count / len(valid)
    print(f"  {n} features: {count:>4} rows ({pct:.1f}%)")

print(f"\nFeatures available per row (counting non-zero non-NaN):")
for n in range(5):
    count = (n_feat_strict == n).sum()
    pct = 100 * count / len(valid)
    print(f"  {n} features: {count:>4} rows ({pct:.1f}%)")

# AUC by feature count
print(f"\nAUC by number of features available:")
for n in range(1, 5):
    mask = n_feat >= n
    sub = valid[mask]
    if len(sub) > 20 and sub['immunogenic'].nunique() == 2:
        auc = roc_auc_score(
            sub['immunogenic'], sub[RL_SCORE])
        print(f"  >= {n} features: AUC={auc:.4f} "
              f"(n={len(sub)}, +={int(sub['immunogenic'].sum())})")

print(f"\n{'='*70}")
print("EDA COMPLETE — paste full output for review")
print(f"{'='*70}")
