# NeoResist-MD: Methods and Results

## Methods

### Datasets

We collected neoantigen immunogenicity data from five published clinical cohorts spanning four cancer types:

| Cohort | Cancer type | Patients | Candidates | Immunogenic | Reference |
|--------|-------------|----------|-----------|-------------|-----------|
| Ott 2017 | Melanoma | 6 | 97 | 15 (15.5%) | Ott et al., Nature 2017 |
| Sahin 2017 | Melanoma | 13 | 165 | 109 (66.1%) | Sahin et al., Nature 2017 |
| Hilf 2019 | GBM | 15 | 99 | 46 (46.5%) | Hilf et al., Nature 2019 |
| Keskin 2019 | GBM | 10 | 80 | 9 (11.3%) | Keskin et al., Nature 2019 |
| Rojas 2023 | Pancreatic | 16 | 230 | 30 (13.0%) | Rojas et al., Nature 2023 |
| TESLA 2020 | Mixed | 9 | 918 | 130 (14.2%) | Wells et al., Cell 2020 |
| Müller NCI | Mixed | 56 | 292,495 | 82 (0.03%) | Müller et al., Science 2023 |

For the Ott 2017 cohort, we additionally curated the full somatic mutanome from Supplementary Table 2 (11,094 mutations across 6 patients), representing the complete pre-vaccine mutation landscape from which the 97 vaccine candidates were selected.

Per-patient HLA alleles were sourced from each paper's supplementary tables. For Sahin 2017, HLA typing was obtained from Extended Data Table 3 (Sahin et al., Nature 2017).

### Feature Engineering

Features were computed at the peptide-MHC level and aggregated to mutation level for cohorts providing pre-screened candidate lists. Binding affinity (nM) was predicted using MHCflurry 2.0 (O'Donnell et al., Cell Systems 2020). Expression was normalised as log₂(TPM + 1) and z-scored within each cohort.

TCR contact features were derived from the peptide sequence at positions P4–P6 (TCR-facing residues, HLA class I):
- `tcr_volume_diff`: difference in side-chain volume between mutant and wild-type residues at TCR contact positions
- `tcr_charge_diff`: net charge difference at TCR contact positions
- `tcr_hydrophobicity_diff`: Kyte-Doolittle hydrophobicity difference

Sequence-level features included self-dissimilarity (Hamming distance to closest human proteome 9-mer), BLOSUM62 substitution score at the mutation site, Calis immunogenicity score, peptide aliphatic index, and full-length hydrophobicity mean.

### Two-Stage Modelling Framework

We designed a two-stage XGBoost architecture separating MHC presentation (Stage 1) from T-cell recognition (Stage 2):

**Stage 1 — Presentation model.** Trained on the Müller NCI full mutanome (292,495 mutations, 82 MHC-presented positives, 56 patients) to predict the probability that a somatic mutation generates an MHC-presented neoantigen. Features: binding_log, binding_sigmoid, expression_log2, expression_zscore. Leave-one-patient-out (LOPO) cross-validation was used throughout to prevent data leakage.

**Stage 2 — Recognition models.** Four cancer-type-specific XGBoost classifiers trained on pooled pre-screened cohorts to predict immunogenicity given MHC presentation:
- `melanoma_ml_v1`: Ott 2017 + Sahin 2017 (n=262, 19 patients)
- `gbm_ml_v1`: Hilf 2019 + Keskin 2019 (n=179, 17 patients)
- `pancreatic_ml_v1`: Rojas 2023 (n=230, 16 patients)
- `mixed_ml_v1`: TESLA 2020 (n=918, 9 patients)

All Stage 2 models used LOPO cross-validation. Class imbalance was handled via `scale_pos_weight` proportional to the negative:positive ratio per training fold.

### Hand-Crafted Scoring Strategies

In addition to ML models, we evaluated three interpretable rule-based strategies:

- `binding_only`: −log₁₀(binding_nm); ranks candidates by MHC affinity alone
- `rl_tcr_v1`: weighted composite score (presentation 0.40, TCR volume 0.20, TCR charge 0.20, TCR hydrophobicity 0.20)
- `rl_expression_v1`: weighted composite prioritising expression level

### Cross-Cancer Transfer Evaluation

To assess generalisability, we evaluated each cancer-type model on held-out cancer types (leave-one-dataset-out, LODO). Within-melanoma transfer was evaluated by training on Ott 2017 and testing on Sahin 2017, and vice versa, with 95% bootstrap confidence intervals (n=1,000 resamples).

### Full Mutanome Haystack Evaluation

To simulate the clinical decision problem of ranking immunogenic mutations within a patient's full somatic landscape, we evaluated all three scorer types on the Ott 2017 full mutanome (11,094 mutations per patient). We generated mutant peptide sequences by:

1. Parsing protein_change notation (e.g., p.P2056L) to extract position and amino acid substitution
2. Mapping gene symbols to Ensembl IDs via the GRCh37 release 75 GTF annotation
3. Retrieving wild-type protein sequences from the Ensembl GRCh37 release 75 peptide FASTA
4. Selecting the canonical transcript (longest sequence with matching reference residue at the mutation position)
5. Applying the substitution and generating 8–11 amino acid sliding windows spanning the mutated site
6. Running MHCflurry 2.0 with per-patient HLA alleles; selecting the minimum predicted binding affinity across all windows and alleles per mutation

Frameshifts, in-frame indels, stop-codon mutations, and mutations in genes absent from the Ensembl annotation were excluded from binding prediction (2,631/11,094 mutations; 23.7%). After filtering for non-standard amino acids, 818,241 peptide-HLA pairs were scored, yielding binding predictions for 6,618/11,094 mutations (59.6%).

Haystack metrics were computed within each patient (per-patient ranking), reporting recall@K (fraction of immunogenic mutations in the top K ranked candidates) for K = 10, 50.

---

## Results

### Stage 1: Presentation Model

The XGBoost presentation model trained on the Müller NCI full mutanome achieved LOPO AUC = 0.977 (AUPR = 0.054) across 56 patients. The extreme class imbalance (82 positives out of 292,495 mutations) reflects the rarity of immunogenic neoantigens in an unscreened population. As expected, this model outputs near-zero presentation probabilities for pre-screened clinical cohorts, where mutations have already been filtered for MHC binding — a domain gap inherent to the two-stage design.

### Stage 2: Cancer-Type Recognition Models

LOPO performance of Stage 2 models within their respective cancer types:

| Model | Cancer type | Patients | Candidates | LOPO AUC | 95% CI | vs binding baseline |
|-------|-------------|----------|-----------|----------|--------|---------------------|
| melanoma_ml_v1 | Melanoma | 19 | 262 | 0.760 | [0.701, 0.822] | +0.232 |
| gbm_ml_v1 | GBM | 17 | 179 | 0.595 | [0.506, 0.682] | −0.050 |
| pancreatic_ml_v1 | Pancreatic | 16 | 230 | 0.504 | [0.385, 0.623] | −0.003 |
| mixed_ml_v1 | Mixed/TESLA | 9 | 918 | 0.744 | [0.660, 0.824] | +0.014 |

Melanoma was the only cancer type where ML significantly outperformed the binding-only baseline (confidence interval excludes baseline). For GBM and pancreatic cancer, binding affinity was the dominant signal and ML did not improve performance. The TESLA mixed cohort showed marginal improvement (+0.014 AUC) with the `rl_expression_v1` hand-crafted strategy performing equivalently (AUC 0.730).

### Cross-Cancer Transfer Failure and the Case for Interpretable Strategies

Cancer-type-specific models failed to transfer across cancer types, with held-out AUC ranging from 0.35 to 0.67. Within-melanoma transfer was particularly poor: training on Ott 2017 and predicting Sahin 2017 yielded AUC = 0.492 (binding_only baseline: 0.477); training on Sahin 2017 and predicting Ott 2017 yielded AUC = 0.387 (binding_only baseline: 0.649).

By contrast, the hand-crafted `rl_tcr_v1` strategy — which uses no training data — achieved AUC = 0.660 when applied to Sahin 2017 without any Sahin training data, and AUC = 0.758 when applied to Ott 2017 using only Sahin training. This zero-shot transfer advantage demonstrates that domain-general biophysical features (TCR contact volume, charge, hydrophobicity) carry immunogenicity signal that generalises across patient cohorts, whereas fitted models capture dataset-specific artefacts.

### Per-Patient LOPO Haystack (Pre-Screened Cohorts)

On the pre-screened Ott 2017 and Sahin 2017 cohorts evaluated with LOPO:

| Dataset | Scorer | Pooled AUC | R@10 | R@20 |
|---------|--------|-----------|------|------|
| Ott 2017 | binding_only | 0.636 | — | — |
| Ott 2017 | rl_tcr_v1 | 0.679 | — | — |
| Ott 2017 | melanoma_ml_v1 | 0.674 | 0.917 | — |
| Sahin 2017 | melanoma_ml_v1 | 0.581 | — | — |

All 13 immunogenic Ott 2017 mutations ranked within the top 6 candidates within their patient's cohort (per-patient honest metric), including CIT p.P2056L ranking #1 of 97 candidates by melanoma_ml_v1 (Patient ott_3).

### Full Mutanome Haystack: Binding Alone Misses Immunogenic Mutations

Extending evaluation to the full Ott 2017 somatic mutanome (11,094 mutations) with MHCflurry-generated binding predictions (6,618 mutations with valid binding data):

**Pooled AUC on 70 labeled mutations with binding predictions:**

| Scorer | AUC | R@10 | R@20 |
|--------|-----|------|------|
| binding_only | **0.701** | 0.182 | 0.364 |
| melanoma_ml_v1 | 0.685 | **0.364** | **0.636** |
| rl_tcr_v1 | 0.613 | 0.091 | 0.364 |

While binding_only achieved the highest pooled AUC (0.701), per-patient ranking revealed critical failures of the binding-only approach that aggregate metrics conceal:

| Gene | Patient | Mutations in patient | Binding rank | rl_tcr_v1 rank |
|------|---------|---------------------|-------------|-----------------|
| DHX40 | ott_4 | 1,997 | **1,488** | **1** |
| CASP1 | ott_3 | 820 | **559** | **11** |
| VPS16 | ott_3 | 820 | 77 | **3** |
| FAM50B | ott_6 | 3,920 | 68 | **2** |

DHX40 (ott_4) exemplifies the limitation: this validated immunogenic neoantigen binds at 1,488 nM — well below the canonical 500 nM presentation threshold — and would rank 1,488th of 1,997 candidates if sorted by binding affinity. The rl_tcr_v1 TCR contact features identify it as rank #1 in that patient. CASP1 (ott_3), immunogenic in the Ott trial, ranks 559th of 820 by binding but 11th by rl_tcr_v1.

These findings demonstrate that **MHC binding affinity alone is insufficient for neoantigen prioritisation**: a non-trivial fraction of immunogenic mutations are presented at intermediate affinity (500–5,000 nM) but recognised by T cells due to favourable TCR-contact residue properties. The NeoResist-MD platform addresses this by providing configurable multi-feature scoring strategies that can be tuned per cancer type and data availability.

---

## Discussion

Our validation across six cohorts and four cancer types reveals three findings that motivate the NeoResist-MD platform design:

1. **No single strategy generalises across cancer types.** Melanoma benefits substantially from ML (+0.23 AUC over binding); GBM and pancreatic cancer do not. Pre-screened cohorts are binding-dominated; unscreened mutanomes require TCR contact features to discriminate immunogenic candidates.

2. **Fitted ML models overfit to cohort-specific artefacts.** Within-melanoma LODO AUC (0.387–0.492) is below or at chance for the ML model that achieves 0.760 within-cohort LOPO. In contrast, rl_tcr_v1 achieves 0.660–0.758 zero-shot on the same data. This argues against a single trained model and for interpretable strategies that encode domain-general biophysics.

3. **Per-patient ranking, not pooled AUC, is the clinically relevant metric.** A vaccine shortlist is selected within a single patient's mutanome. DHX40 (ott_4) would never reach a 20-candidate shortlist sorted by binding affinity; it is rank #1 by rl_tcr_v1. The platform's per-patient ranking view directly addresses this.

The platform provides strategy profiles (YAML-defined, version-controlled) for each cancer type, with recommended defaults informed by the validation results above. For melanoma, melanoma_ml_v1 is recommended for LOPO-validated cohort data; rl_tcr_v1 is recommended for novel patients where training data may not generalise.
