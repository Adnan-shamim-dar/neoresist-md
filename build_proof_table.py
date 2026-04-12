import pandas as pd
import json
from pathlib import Path

infile = Path("C:/Users/rambe/Documents/neovax/data/final/enriched_candidates.parquet")
outdir = Path("C:/Users/rambe/Documents/neovax/data/final")
df = pd.read_parquet(infile)

for col in ["purity_value_used","real_expression_tpm","real_ccf","priority_score"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

summary = df.groupby("patient_id", dropna=False).agg(
    candidate_rows=("patient_id", "size"),
    unique_peptides=("peptide", pd.Series.nunique),
    mean_purity=("purity_value_used", "mean"),
    median_purity=("purity_value_used", "median"),
    mean_real_expression=("real_expression_tpm", "mean"),
    mean_real_ccf=("real_ccf", "mean"),
    max_priority_score=("priority_score", "max"),
    unresolved_purity_rows=("purity_source_used", lambda s: (s == "unresolved").sum()),
    tcga_purity_rows=("purity_source_used", lambda s: (s == "tcga_metadata").sum()),
    user_purity_rows=("purity_source_used", lambda s: (s == "user_file").sum()),
    stub_purity_rows=("purity_source_used", lambda s: (s == "stub_fallback").sum()),
    expr_sources=("expression_source", lambda s: s.mode().iat[0] if not s.mode().empty else None),
    ccf_sources=("ccf_source", lambda s: s.mode().iat[0] if not s.mode().empty else None),
    purity_sources=("purity_source_used", lambda s: s.mode().iat[0] if not s.mode().empty else None),
    resolution_status=("resolution_status", lambda s: s.mode().iat[0] if not s.mode().empty else None),
).reset_index()

summary["purity_source_breakdown"] = summary.apply(lambda r: json.dumps({
    "tcga_metadata": int(r["tcga_purity_rows"]),
    "user_file": int(r["user_purity_rows"]),
    "stub_fallback": int(r["stub_purity_rows"]),
    "unresolved": int(r["unresolved_purity_rows"])
}, sort_keys=True), axis=1)

summary = summary.sort_values(["candidate_rows","mean_purity"], ascending=[False, False])
summary.to_csv(outdir / "proof_table.csv", index=False)
print(outdir / "proof_table.csv")
print(summary.head(10).to_string(index=False))
