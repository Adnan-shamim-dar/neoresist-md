import os
import subprocess
from pathlib import Path

# Find all sarc files
files = sorted(Path("data").glob("sarc_*.tsv"))
print(f"Found {len(files)} SARC patients")

for i, maf in enumerate(files):
    patient = maf.stem
    outdir = f"data/neovax_runs/{patient}"
    Path(outdir).mkdir(exist_ok=True)
    
    print(f"[{i+1}/{len(files)}] {patient}")
    
    cmd = [
        "python", "-m", "backend.api.cli.neovax_cli",
        "--input", str(maf),
        "--input-mode", "maf",
        "--hla", "HLA-A*02:01",
        "--output-dir", outdir
    ]
    
    result = subprocess.run(cmd)
    print(f"Exit code: {result.returncode}")
    
print("\n✅ Cohort processing complete")