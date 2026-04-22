import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

tm = pd.read_csv(
    'backend/validation/artifacts/training_matrix.csv')
valid = tm[tm['immunogenic'].isin([0, 1])].copy()

print("=" * 70)
print("PRESENTATION SCORE INVERSION DIAGNOSIS")
print("=" * 70)

# 1. Compare baseline vs presentation score
print("\n1. SCORE COMPARISON")
print(f"binding_affinity_baseline_score AUC: "
      f"{roc_auc_score(valid['immunogenic'], valid['binding_affinity_baseline_score']):.4f}")
print(f"presentation_score AUC: "
      f"{roc_auc_score(valid['immunogenic'], valid['presentation_score']):.4f}")
print(f"1 - presentation_score AUC: "
      f"{roc_auc_score(valid['immunogenic'], 1 - valid['presentation_score']):.4f}")

# 2. Check correlation between the two
corr = valid['binding_affinity_baseline_score'].corr(
    valid['presentation_score'])
print(f"\nCorrelation between baseline and presentation: "
      f"{corr:.4f}")

# 3. Print first 20 rows comparing the two scores
print("\n2. SIDE-BY-SIDE COMPARISON (first 20 rows)")
print(f"{'binding_nm':>12} {'baseline':>10} {'pres_score':>10} "
      f"{'immuno':>6}")
for _, row in valid.head(20).iterrows():
    print(f"{row['binding_affinity_nm']:>12} "
          f"{row['binding_affinity_baseline_score']:>10.4f} "
          f"{row['presentation_score']:>10.4f} "
          f"{int(row['immunogenic']):>6}")

# 4. Check direction: does lower nM = higher score?
print("\n3. DIRECTION CHECK")
has_nm = valid[valid['binding_affinity_nm_numeric'].notna()].copy()
if len(has_nm) > 20:
    # For rows with actual nM values
    # Strong binders (low nM) should have HIGH scores
    strong = has_nm[has_nm['binding_affinity_nm_numeric'] < 50]
    weak = has_nm[has_nm['binding_affinity_nm_numeric'] > 500]
    
    print(f"Strong binders (<50nM): n={len(strong)}")
    if len(strong) > 0:
        print(f"  Mean baseline: "
              f"{strong['binding_affinity_baseline_score'].mean():.4f}")
        print(f"  Mean presentation: "
              f"{strong['presentation_score'].mean():.4f}")
    
    print(f"Weak binders (>500nM): n={len(weak)}")
    if len(weak) > 0:
        print(f"  Mean baseline: "
              f"{weak['binding_affinity_baseline_score'].mean():.4f}")
        print(f"  Mean presentation: "
              f"{weak['presentation_score'].mean():.4f}")
    
    print(f"\nExpected: strong binders should have HIGHER "
          f"scores than weak binders")
    if len(strong) > 0 and len(weak) > 0:
        bl_correct = strong['binding_affinity_baseline_score'].mean() > \
                     weak['binding_affinity_baseline_score'].mean()
        pr_correct = strong['presentation_score'].mean() > \
                     weak['presentation_score'].mean()
        print(f"  Baseline direction correct: {bl_correct}")
        print(f"  Presentation direction correct: {pr_correct}")

# 5. Check what presentation_score actually contains
print("\n4. PRESENTATION SCORE DISTRIBUTION")
print(valid['presentation_score'].describe())
print(f"\nBinding source breakdown:")
if 'binding_source' in valid.columns:
    print(valid['binding_source'].value_counts())

# 6. Check if MHCflurry predictions are inverted
print("\n5. MHCFLURRY vs REPORTED AFFINITY")
if 'predicted_affinity_nm' in valid.columns:
    has_both = valid[
        valid['binding_affinity_nm_numeric'].notna() & 
        valid['predicted_affinity_nm'].notna()].copy()
    if len(has_both) > 10:
        corr = has_both['binding_affinity_nm_numeric'].corr(
            has_both['predicted_affinity_nm'])
        print(f"Correlation reported vs predicted nM: "
              f"{corr:.4f}")
        print(f"First 10 comparisons:")
        for _, row in has_both.head(10).iterrows():
            print(f"  reported={row['binding_affinity_nm_numeric']:.1f}nM "
                  f"predicted={row['predicted_affinity_nm']:.1f}nM "
                  f"pres_score={row['presentation_score']:.4f} "
                  f"baseline={row['binding_affinity_baseline_score']:.4f}")

# 7. How is binding_affinity_baseline_score computed?
print("\n6. BASELINE SCORE FORMULA CHECK")
has_nm = valid[valid['binding_affinity_nm_numeric'].notna()].copy()
if len(has_nm) > 10:
    # Check if baseline = 1 - log(nM)/log(50000)
    has_nm['expected_baseline'] = 1 - (
        np.log10(has_nm['binding_affinity_nm_numeric']) / 
        np.log10(50000))
    has_nm['expected_baseline'] = has_nm[
        'expected_baseline'].clip(0, 1)
    
    corr = has_nm['binding_affinity_baseline_score'].corr(
        has_nm['expected_baseline'])
    print(f"Correlation baseline vs expected formula: "
          f"{corr:.4f}")
    
    # Check another formula: 1 - min(nM/50000, 1)
    has_nm['expected_linear'] = 1 - (
        has_nm['binding_affinity_nm_numeric'] / 50000).clip(0, 1)
    corr2 = has_nm['binding_affinity_baseline_score'].corr(
        has_nm['expected_linear'])
    print(f"Correlation baseline vs linear: {corr2:.4f}")

# 8. Per-cancer presentation check
print("\n7. PER-CANCER PRESENTATION DIRECTION")
for cancer in valid['cancer_type'].unique():
    ct = valid[valid['cancer_type'] == cancer]
    if ct['immunogenic'].nunique() == 2:
        auc_bl = roc_auc_score(
            ct['immunogenic'], 
            ct['binding_affinity_baseline_score'])
        auc_pr = roc_auc_score(
            ct['immunogenic'], ct['presentation_score'])
        auc_inv = roc_auc_score(
            ct['immunogenic'], 1 - ct['presentation_score'])
        print(f"\n{cancer}:")
        print(f"  Baseline AUC:      {auc_bl:.4f}")
        print(f"  Presentation AUC:  {auc_pr:.4f}")
        print(f"  1-Presentation AUC:{auc_inv:.4f}")

# 9. THE FIX TEST
# What happens if we use baseline_score in place of 
# presentation_score in the RL formula?
print("\n" + "=" * 70)
print("8. SIMULATED FIX: Use baseline as presentation")
print("=" * 70)

FEATURES_FIXED = ['expression_score', 
                   'binding_affinity_baseline_score',
                   'ccf', 'self_dissimilarity']

X_fixed = valid[FEATURES_FIXED].fillna(0).values
y = valid['immunogenic'].values

# Try the grid search best weights
from itertools import product as iprod
best_auc = 0
best_w = None
steps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for w0, w1, w2, w3 in iprod(steps, steps, steps, steps):
    t = w0+w1+w2+w3
    if t == 0: continue
    ws = np.array([w0,w1,w2,w3]) / t
    score = (X_fixed * ws).sum(axis=1)
    auc = roc_auc_score(y, score)
    if auc > best_auc:
        best_auc = auc
        best_w = ws

print(f"\nWith baseline replacing presentation:")
print(f"  Best AUC: {best_auc:.4f}")
print(f"  Weights: expr={best_w[0]:.3f} "
      f"pres(baseline)={best_w[1]:.3f} "
      f"ccf={best_w[2]:.3f} dissim={best_w[3]:.3f}")

# Per cancer with fix
print(f"\nPer-cancer with baseline as presentation:")
for cancer in valid['cancer_type'].unique():
    ct = valid[valid['cancer_type'] == cancer]
    if len(ct) < 20 or ct['immunogenic'].nunique() < 2:
        continue
    X_ct = ct[FEATURES_FIXED].fillna(0).values
    y_ct = ct['immunogenic'].values
    
    best_ct = 0
    best_ct_w = None
    coarse = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for w0,w1,w2,w3 in iprod(coarse,coarse,coarse,coarse):
        t = w0+w1+w2+w3
        if t == 0: continue
        ws = np.array([w0,w1,w2,w3]) / t
        sc = (X_ct * ws).sum(axis=1)
        a = roc_auc_score(y_ct, sc)
        if a > best_ct:
            best_ct = a
            best_ct_w = ws
    print(f"  {cancer}: AUC={best_ct:.4f} "
          f"(expr={best_ct_w[0]:.2f} "
          f"pres={best_ct_w[1]:.2f} "
          f"ccf={best_ct_w[2]:.2f} "
          f"dissim={best_ct_w[3]:.2f})")

# Logistic regression with fix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_s = scaler.fit_transform(X_fixed)
lr = LogisticRegression(C=1.0, class_weight='balanced',
                         max_iter=1000, random_state=42)
lr.fit(X_s, y)
lr_prob = lr.predict_proba(X_s)[:, 1]
auc_lr_fixed = roc_auc_score(y, lr_prob)

print(f"\nLogistic regression with fix: AUC={auc_lr_fixed:.4f}")
print(f"Coefficients:")
feat_names = ['expression', 'presentation(baseline)', 
              'ccf', 'self_dissimilarity']
for n, c in zip(feat_names, lr.coef_[0]):
    direction = "+" if c > 0 else "NEGATIVE"
    print(f"  {n:<25} {c:+.4f} [{direction}]")

print(f"\n{'='*70}")
print("DIAGNOSIS COMPLETE")
print(f"{'='*70}")
