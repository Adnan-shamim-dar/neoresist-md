import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

tm = pd.read_csv('backend/validation/artifacts/training_matrix.csv')

# Ensure correct types
for col in ['immunogenic', 'rl_priority_v1', 
            'presentation_only_score', 'expression_score',
            'presentation_score', 'ccf_score', 
            'self_dissimilarity_score']:
    if col in tm.columns:
        tm[col] = pd.to_numeric(tm[col], errors='coerce')

# Filter to usable with valid labels
if 'usable_for_training' in tm.columns:
    valid = tm[tm['usable_for_training'].astype(str).isin(
        ['1', '1.0', 'True'])].copy()
else:
    valid = tm.copy()

valid = valid[valid['immunogenic'].isin([0, 1])].copy()

print("=" * 60)
print("NEORESIST-MD FULL DIAGNOSTIC REPORT")
print("=" * 60)

print(f"\nTotal rows in matrix: {len(tm)}")
print(f"Usable rows with valid labels: {len(valid)}")
print(f"  immunogenic=1: {(valid['immunogenic']==1).sum()}")
print(f"  immunogenic=0: {(valid['immunogenic']==0).sum()}")

# ============================================
print(f"\n{'='*60}")
print("1. COLUMN INVENTORY")
print(f"{'='*60}")
print(f"Columns: {list(tm.columns)}")

# ============================================
print(f"\n{'='*60}")
print("2. FEATURE COVERAGE PER PAPER")
print(f"{'='*60}")
print(f"{'Paper':<15} {'N':>5} {'expr':>6} {'pres':>6} "
      f"{'ccf':>5} {'dissim':>7} {'rl>0':>6}")
print("-" * 55)
for paper in sorted(valid['paper_source'].unique()):
    sub = valid[valid['paper_source'] == paper]
    n = len(sub)
    e = sub['expression_score'].notna().sum()
    p = sub['presentation_score'].notna().sum()
    c = sub['ccf_score'].notna().sum()
    d = sub['self_dissimilarity_score'].notna().sum()
    r = (sub['rl_priority_v1'] > 0).sum()
    print(f"{paper:<15} {n:>5} {e:>5}  {p:>5}  "
          f"{c:>4}  {d:>6}  {r:>5}")

# ============================================
print(f"\n{'='*60}")
print("3. SCORE DISTRIBUTIONS")
print(f"{'='*60}")
for col in ['rl_priority_v1', 'presentation_only_score',
            'expression_score', 'presentation_score',
            'ccf_score', 'self_dissimilarity_score']:
    if col not in valid.columns:
        print(f"\n{col}: COLUMN MISSING")
        continue
    vals = valid[col].dropna()
    if len(vals) > 0:
        print(f"\n{col} (n={len(vals)}):")
        print(f"  min={vals.min():.4f}  25th={vals.quantile(0.25):.4f}  "
              f"median={vals.median():.4f}  75th={vals.quantile(0.75):.4f}  "
              f"max={vals.max():.4f}")
    else:
        print(f"\n{col}: ALL NaN — NOT WORKING")

# ============================================
print(f"\n{'='*60}")
print("4. IMMUNOGENIC vs NON-IMMUNOGENIC SEPARATION")
print(f"{'='*60}")
immuno = valid[valid['immunogenic'] == 1]
non_immuno = valid[valid['immunogenic'] == 0]

for col in ['rl_priority_v1', 'presentation_only_score',
            'expression_score', 'presentation_score',
            'ccf_score', 'self_dissimilarity_score']:
    if col not in valid.columns:
        continue
    i_vals = immuno[col].dropna()
    n_vals = non_immuno[col].dropna()
    if len(i_vals) > 3 and len(n_vals) > 3:
        diff = i_vals.mean() - n_vals.mean()
        direction = "CORRECT" if diff > 0 else "WRONG"
        print(f"\n{col}:")
        print(f"  Immunogenic:     mean={i_vals.mean():.4f} "
              f"median={i_vals.median():.4f} (n={len(i_vals)})")
        print(f"  Non-immunogenic: mean={n_vals.mean():.4f} "
              f"median={n_vals.median():.4f} (n={len(n_vals)})")
        print(f"  Delta: {diff:+.4f} [{direction}]")

# ============================================
print(f"\n{'='*60}")
print("5. AUC-ROC RESULTS")
print(f"{'='*60}")

# Overall
rl_valid = valid.dropna(subset=['rl_priority_v1'])
if len(rl_valid) > 20 and rl_valid['immunogenic'].nunique() == 2:
    auc_rl_all = roc_auc_score(
        rl_valid['immunogenic'], rl_valid['rl_priority_v1'])
    print(f"\nALL PAPERS (n={len(rl_valid)}):")
    print(f"  RL AUC: {auc_rl_all:.4f}")

pres_valid = valid.dropna(subset=['presentation_only_score'])
pres_valid_nonzero = pres_valid[
    pres_valid['presentation_only_score'] > 0]
if len(pres_valid_nonzero) > 20 and \
   pres_valid_nonzero['immunogenic'].nunique() == 2:
    auc_base_all = roc_auc_score(
        pres_valid_nonzero['immunogenic'],
        pres_valid_nonzero['presentation_only_score'])
    print(f"  Baseline AUC: {auc_base_all:.4f}")
    print(f"  Delta: {auc_rl_all - auc_base_all:+.4f}")

# Per paper
print(f"\nPER-PAPER:")
for paper in sorted(rl_valid['paper_source'].unique()):
    sub = rl_valid[rl_valid['paper_source'] == paper]
    n_pos = (sub['immunogenic'] == 1).sum()
    n_neg = (sub['immunogenic'] == 0).sum()
    if len(sub) > 10 and sub['immunogenic'].nunique() == 2:
        auc = roc_auc_score(
            sub['immunogenic'], sub['rl_priority_v1'])
        
        # Also compute baseline for this paper
        sub_pres = sub.dropna(subset=['presentation_only_score'])
        sub_pres = sub_pres[sub_pres['presentation_only_score'] > 0]
        if len(sub_pres) > 10 and \
           sub_pres['immunogenic'].nunique() == 2:
            auc_b = roc_auc_score(
                sub_pres['immunogenic'],
                sub_pres['presentation_only_score'])
            delta = auc - auc_b
            print(f"  {paper:<15} RL={auc:.4f}  "
                  f"Base={auc_b:.4f}  Delta={delta:+.4f}  "
                  f"(n={len(sub)}, +={n_pos}, -={n_neg})")
        else:
            print(f"  {paper:<15} RL={auc:.4f}  "
                  f"Base=N/A  "
                  f"(n={len(sub)}, +={n_pos}, -={n_neg})")
    else:
        print(f"  {paper:<15} SKIP "
              f"(n={len(sub)}, +={n_pos}, -={n_neg})")

# Ott only
ott = rl_valid[rl_valid['paper_source'] == 'ott_2017']
if len(ott) > 10 and ott['immunogenic'].nunique() == 2:
    auc_ott = roc_auc_score(
        ott['immunogenic'], ott['rl_priority_v1'])
    print(f"\nOTT-ONLY DETAIL:")
    print(f"  RL AUC: {auc_ott:.4f}")
    print(f"  N features available distribution:")
    if 'n_features_available' in ott.columns:
        print(ott['n_features_available'].value_counts()
              .sort_index().to_string())

# ============================================
print(f"\n{'='*60}")
print("6. FEATURE-RICH vs FEATURE-POOR")
print(f"{'='*60}")
if 'n_features_available' in rl_valid.columns:
    for threshold in [1, 2, 3]:
        rich = rl_valid[
            rl_valid['n_features_available'] >= threshold]
        poor = rl_valid[
            rl_valid['n_features_available'] < threshold]
        
        auc_r = "N/A"
        auc_p = "N/A"
        if len(rich) > 20 and rich['immunogenic'].nunique() == 2:
            auc_r = f"{roc_auc_score(rich['immunogenic'], rich['rl_priority_v1']):.4f}"
        if len(poor) > 20 and poor['immunogenic'].nunique() == 2:
            auc_p = f"{roc_auc_score(poor['immunogenic'], poor['rl_priority_v1']):.4f}"
        
        print(f"  >= {threshold} features: AUC={auc_r} "
              f"(n={len(rich)})")
        print(f"  <  {threshold} features: AUC={auc_p} "
              f"(n={len(poor)})")
        print()
else:
    print("  n_features_available column not found")

# ============================================
print(f"\n{'='*60}")
print("7. SHANK2 CASE STUDY")
print(f"{'='*60}")
shank = valid[valid['gene'].astype(str).str.contains(
    'SHANK', case=False, na=False)]
if len(shank) > 0:
    for _, row in shank.iterrows():
        print(f"  Patient: {row['patient_id']}")
        print(f"  Gene: {row['gene']}")
        print(f"  Protein change: {row.get('protein_change','?')}")
        print(f"  Immunogenic: {int(row['immunogenic'])}")
        rl = row['rl_priority_v1']
        bl = row['presentation_only_score']
        print(f"  RL score: {rl:.4f}" if pd.notna(rl) 
              else "  RL score: NaN")
        print(f"  Baseline: {bl:.4f}" if pd.notna(bl) 
              else "  Baseline: NaN")
        print(f"  Expression: {row.get('expression_score','?')}")
        print(f"  Presentation: {row.get('presentation_score','?')}")
        print(f"  Dissimilarity: {row.get('self_dissimilarity_score','?')}")
        print(f"  N features: {row.get('n_features_available','?')}")
else:
    print("  No SHANK rows found")
    # Search more broadly
    keskin = valid[valid['paper_source'].str.contains(
        'keskin', case=False, na=False)]
    print(f"  Keskin rows total: {len(keskin)}")
    if len(keskin) > 0:
        print(f"  Keskin genes: {keskin['gene'].unique().tolist()}")
        print(f"  Keskin immunogenic genes: "
              f"{keskin[keskin['immunogenic']==1]['gene'].unique().tolist()}")

# ============================================
print(f"\n{'='*60}")
print("8. PRESENTATION METHOD BREAKDOWN")
print(f"{'='*60}")
if 'presentation_method' in valid.columns:
    print(valid['presentation_method'].value_counts()
          .to_string())
else:
    print("  Column not in output")

# ============================================
print(f"\n{'='*60}")
print("9. SAMPLE ROWS — TOP 5 IMMUNOGENIC BY RL")
print(f"{'='*60}")
top5 = immuno.nlargest(5, 'rl_priority_v1')
for _, row in top5.iterrows():
    print(f"  {row['paper_source']} | {row['patient_id']} | "
          f"{row['gene']} | RL={row['rl_priority_v1']:.4f} | "
          f"expr={row['expression_score']:.3f}" 
          if pd.notna(row['expression_score']) 
          else f"  {row['paper_source']} | {row['patient_id']} | "
               f"{row['gene']} | RL={row['rl_priority_v1']:.4f} | "
               f"expr=NaN",
          f"| pres={row['presentation_score']:.3f}" 
          if pd.notna(row.get('presentation_score', np.nan)) 
          else "| pres=NaN")

# ============================================
print(f"\n{'='*60}")
print("10. IMMUNOGENICITY BLEND DECOMPOSITION")
print(f"{'='*60}")
# Check if the RL score is actually using expression
# by looking at correlation between components and RL
for col in ['expression_score', 'presentation_score',
            'self_dissimilarity_score', 'ccf_score']:
    if col in valid.columns:
        both = valid[[col, 'rl_priority_v1']].dropna()
        if len(both) > 20:
            corr = both[col].corr(both['rl_priority_v1'])
            print(f"  {col} ↔ rl_priority: r={corr:.4f} "
                  f"(n={len(both)})")

# ============================================
print(f"\n{'='*60}")
print("DONE — PASTE THIS ENTIRE OUTPUT")
print(f"{'='*60}")
