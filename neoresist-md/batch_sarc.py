import subprocess, os
from pathlib import Path
import pandas as pd

cases = [f"data/tcga_sarc_case{i}.tsv" for i in range(1,7)]
proof_table = []

for case_file in cases:
    case_id = Path(case_file).stem
    outdir = f"data/neovax_runs/{case_id}"
    os.makedirs(outdir, exist_ok=True)
    
    cmd = [
        "python", "-m", "backend.api.cli.neovax_cli",
        "--input", case_file, "--input-mode", "maf",
        "--hla", "HLA-A*02:01", "--output-dir", outdir
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    status = "SUCCESS" if result.returncode == 0 else "FAILED"
    
    # Parse metrics
    if status == "SUCCESS" and os.path.exists(f"{outdir}/case_report.md"):
        with open(f"{outdir}/case_report.md") as f:
            lines = f.readlines()
            n_input = int([l for l in lines if "number_of_input_rows:" in l][0].split(": ")[1])
            n_output = int([l for l in lines if "number_of_output_candidates:" in l][0].split(": ")[1])
        
        proof_table.append({
            "case_id": case_id, "status": status,
            "input_mutations": n_input, "output_candidates": n_output,
            "fanout": round(n_output/n_input, 1)
        })
        print(f"✅ {case_id}: {n_input}→{n_output} ({n_output/n_input:.1f}x)")
    
    else:
        print(f"❌ {case_id}: {status}")

pd.DataFrame(proof_table).to_csv("data/sarc_cohort_proof.csv", index=False)
print(f"\n🏆 SAVED: data/sarc_cohort_proof.csv ({len(proof_table)} cases)")
