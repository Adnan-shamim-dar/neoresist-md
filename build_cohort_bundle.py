import json
from pathlib import Path
import pandas as pd

BASE = Path("C:/Users/rambe/Documents/neovax")
INFILE = BASE / "data/final/enriched_candidates.parquet"
OUTDIR = BASE / "data/final"
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(INFILE)
for col in ["purity_value_used", "real_expression_tpm", "real_ccf", "priority_score"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

patient = df.groupby("patient_id", dropna=False).agg(
    candidate_rows=("patient_id", "size"),
    unique_peptides=("peptide", pd.Series.nunique),
    unique_genes=("gene_name", pd.Series.nunique),
    mean_purity=("purity_value_used", "mean"),
    median_purity=("purity_value_used", "median"),
    mean_real_expression=("real_expression_tpm", "mean"),
    mean_real_ccf=("real_ccf", "mean"),
    max_priority_score=("priority_score", "max"),
    tcga_purity_rows=("purity_source_used", lambda s: (s == "tcga_metadata").sum()),
    user_purity_rows=("purity_source_used", lambda s: (s == "user_file").sum()),
    stub_purity_rows=("purity_source_used", lambda s: (s == "stub_fallback").sum()),
    unresolved_purity_rows=("purity_source_used", lambda s: (s == "unresolved").sum()),
    resolution_status=("resolution_status", lambda s: s.mode().iat[0] if not s.mode().empty else None),
    expression_source=("expression_source", lambda s: s.mode().iat[0] if not s.mode().empty else None),
    ccf_source=("ccf_source", lambda s: s.mode().iat[0] if not s.mode().empty else None),
    purity_source=("purity_source_used", lambda s: s.mode().iat[0] if not s.mode().empty else None),
).reset_index()

patient["purity_source_breakdown"] = patient.apply(lambda r: json.dumps({
    "tcga_metadata": int(r["tcga_purity_rows"]),
    "user_file": int(r["user_purity_rows"]),
    "stub_fallback": int(r["stub_purity_rows"]),
    "unresolved": int(r["unresolved_purity_rows"])
}, sort_keys=True), axis=1)

patient = patient.sort_values(["candidate_rows", "unique_peptides"], ascending=[False, False])
patient.to_csv(OUTDIR / "proof_table.csv", index=False)

per_patient = df.groupby("patient_id", dropna=False).agg(
    best_priority_score=("priority_score", "max"),
    best_presentation_score=("presentation_score", "max"),
    best_affinity_nm=("affinity_nm", "min"),
    best_percentile_rank=("percentile_rank", "min"),
    peptides_with_high_tier=("tier", lambda s: (s.astype(str).str.lower() == "high").sum()),
    peptides_with_medium_tier=("tier", lambda s: (s.astype(str).str.lower() == "medium").sum()),
    peptides_with_low_tier=("tier", lambda s: (s.astype(str).str.lower() == "low").sum()),
).reset_index()
per_patient.to_csv(OUTDIR / "patient_metrics.csv", index=False)

summary = {
    "input_rows": int(len(df)),
    "unique_patients": int(df["patient_id"].nunique(dropna=False)),
    "output_rows": int(len(patient)),
    "proof_table": str((OUTDIR / "proof_table.csv").as_posix()),
    "patient_metrics": str((OUTDIR / "patient_metrics.csv").as_posix()),
}
(OUTDIR / "cohort_bundle_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print(patient.head(10).to_string(index=False))
