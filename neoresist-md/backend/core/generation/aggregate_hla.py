from pathlib import Path
import pandas as pd

BASE = Path("data/neovaxruns/hla_test")
rows = []

for hla_dir in BASE.iterdir():
    if not hla_dir.is_dir(): continue
    hla = hla_dir.name  # e.g. HLA-A0101
    for patient_dir in hla_dir.iterdir():
        report = patient_dir / "casereport.md"
        if not report.exists(): continue
        content = report.read_text()
        try:
            nin  = int([l for l in content.splitlines() if "numberofinputrows" in l][0].split()[-1])
            nout = int([l for l in content.splitlines() if "numberofoutputcandidates" in l][0].split()[-1])
            rows.append({
                "patient":    patient_dir.name,
                "hla_allele": hla,
                "mutations":  nin,
                "candidates": nout,
                "fanout":     round(nout / nin, 1) if nin > 0 else 0
            })
        except: continue

df = pd.DataFrame(rows)
df.to_csv("data/multi_hla_cohort.csv", index=False)
print(f"Done: {len(df)} rows, {df['hla_allele'].nunique()} alleles, {df['patient'].nunique()} patients")
print(df.groupby("hla_allele")["candidates"].sum())
