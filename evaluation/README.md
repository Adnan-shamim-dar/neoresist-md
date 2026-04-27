# evaluation/

Scripts to reproduce all quantitative results in the paper.

## Reproduce held-out melanoma result (Table 3)

```bash
python evaluation/sahin_held_out_eval.py
```

Expected output: mean patient AUC = 0.575, delta = +0.105, p = 0.011

## Reproduce Borch/IMPROVE significance test (Table 5)

```bash
python evaluation/borch_significance_test.py
```

Expected output: paired permutation p = 0.067, bootstrap CI = [−0.028, +0.123]

## Generate all paper figures

```bash
python evaluation/generate_figures.py
```

Output: `figures/fig1_framework.png`, `fig2_melanoma.png`, `fig3_borch.png`, `fig4_transfer.png`
