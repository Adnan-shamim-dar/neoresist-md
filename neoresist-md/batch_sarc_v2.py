#!/usr/bin/env python3
"""
NeoResist-MD: TCGA-SARC Cohort NeoVax Pipeline v2.0
Processes 238 cases → proof table w/ stats
"""

import subprocess
import os
import pandas as pd
from pathlib import Path
import time

def parse_case_report(report_path):
    """Extract metrics from case_report.md"""
    with open(report_path, 'r') as f:
        lines = f.readlines()
    n_input = int(next(l for l in lines if 'number_of_input_rows:' in l).split(': ')[1])
    n_output = int(next(l for l in lines if 'number_of_output_candidates:' in l).split(': ')[1])
    return n_input, n_output

# Cohort config
COHORT_SIZE = 238
HLA = "HLA-A*02:01"
BASE_DIR = Path("data")
OUTPUT_DIR = BASE_DIR / "neovax_runs"
PROOF_FILE = BASE_DIR / "sarc_cohort_proof_v2.csv"

print(f"🚀 NeoResist-MD SARC Cohort Pipeline (HLA: {HLA})")
print(f"Target: {COHORT_SIZE} cases")

proof_table = []
processed = 0

for i in range(1, COHORT_SIZE + 1):
    case_file = BASE_DIR / f"tcga_sarc_case{str(i).zfill(3)}.tsv"
    
    if not case_file.exists():
        print(f"⏭️  case{str(i).zfill(3)}: Missing TSV")
        continue
    
    case_id = case_file.stem
    outdir = OUTPUT_DIR / case_id
    outdir.mkdir(exist_ok=True)
    
    cmd = [
        "python", "-m", "backend.api.cli.neovax_cli",
        "--input", str(case_file), "--input-mode", "maf",
        "--hla", HLA, "--output-dir", str(outdir)
    ]
    
    print(f"[{i:3d}/{COHORT_SIZE}] Processing {case_id}...", end=" ")
    start_time = time.time()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0 and (outdir / "case_report.md").exists():
        try:
            n_input, n_output = parse_case_report(outdir / "case_report.md")
            fanout = round(n_output / n_input, 1)
            runtime = round(time.time() - start_time, 1)
            
            proof_table.append({
                "case_id": case_id,
                "sample_barcode": case_id,  # Update from manifest later
                "mutations": n_input,
                "candidates": n_output,
                "fanout": fanout,
                "runtime_s": runtime,
                "hla": HLA,
                "status": "SUCCESS"
            })
            
            print(f"✅ {n_input}→{n_output} ({fanout}x, {runtime}s)")
            processed += 1
            
        except Exception as e:
            print(f"❌ Parse error: {e}")
            proof_table.append({"case_id": case_id, "status": "PARSE_ERROR"})
    else:
        print(f"❌ FAILED")
        proof_table.append({"case_id": case_id, "status": "FAILED"})

# Save + stats
df = pd.DataFrame(proof_table)
df.to_csv(PROOF_FILE, index=False)

success_cases = df[df['status'] == 'SUCCESS']
if len(success_cases) > 0:
    print(f"\n🏆 COHORT COMPLETE:")
    print(f"  Processed: {processed}/{COHORT_SIZE}")
    print(f"  Success:   {len(success_cases)}")
    print(f"  Mutations: {success_cases['mutations'].sum():,.0f}")
    print(f"  Candidates:{success_cases['candidates'].sum():,.0f}")
    print(f"  Mean fanout: {success_cases['fanout'].mean():.1f}x")
    print(f"\n📊 Saved: {PROOF_FILE}")
else:
    print("\n❌ No successful cases")

print("\nNext: python batch_sarc_v2.py --resume")  # Future enhancement