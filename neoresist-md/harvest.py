import pandas as pd
import os

proof = []
for d in os.listdir("data/neovax_runs"):
    report = f"data/neovax_runs/{d}/case_report.md"
    if os.path.exists(report):
        with open(report) as f:
            lines = f.readlines()
            inp = int([l for l in lines if "number_of_input_rows:" in l][0].split(": ")[1])
            out = int([l for l in lines if "number_of_output_candidates:" in l][0].split(": ")[1])
        proof.append({"patient": d, "mutations": inp, "candidates": out, "fanout": round(out/inp,1)})

df = pd.DataFrame(proof)
df.to_csv("data/final_sarc_cohort.csv", index=False)
print(df.describe())
print(f"\nSAVED: data/final_sarc_cohort.csv ({len(df)} patients)")