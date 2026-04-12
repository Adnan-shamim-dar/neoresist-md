import pandas as pd
import subprocess
from pathlib import Path

manifest = pd.read_csv("cases_manifest.csv")
outdir = Path("data/final/batch_runs")
outdir.mkdir(parents=True, exist_ok=True)

for _, row in manifest.iterrows():
    patient_id = str(row["patient_id"])
    input_path = str(row["input_path"])
    run_dir = outdir / patient_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "backend.cli.enrich_cohort",
        "--stub",
        "--purity-file",
        "manual_purity.csv" if Path("manual_purity.csv").exists() else "",
    ]
    cmd = [c for c in cmd if c]
    subprocess.run(cmd, check=True)

print("done")