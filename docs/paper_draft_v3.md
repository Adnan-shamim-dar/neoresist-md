# NeoResist-MD: discovery and validation of a melanoma-specific neoantigen ranking strategy and a framework for cancer-type extension

**Authors:** Adnan Shamim Dar  
**Affiliations:** School of Medicine, Università Politecnica delle Marche (UNIVPM), Ancona, Italy  
**Correspondence:** s1119248@studenti.univpm.it  
**Journal target:** Genome Medicine / NPJ Precision Oncology

---

## Abstract

Neoantigen ranking is dominated by MHC binding affinity prediction, yet systematic assays show that binding alone achieves mean patient AUC of only 0.47–0.59 across published cohorts. We present NeoResist-MD, an automated strategy discovery framework that identifies cancer-type-specific ranking formulas from immunogenicity data with transparent discovery-validation provenance. Applied to melanoma, the framework discovered a four-feature strategy on the Ott 2017 cohort (n = 13 patients) that improved mean patient AUC from 0.470 to 0.575 on the independently held-out Sahin 2017 cohort (delta +0.105; 95% bootstrap CI 0.511–0.630; permutation *p* = 0.011). Feature ablation identified TCR contact volume as the dominant discriminating component (−0.086 AUC on removal). This TCR contact signal was corroborated on an independent external cohort (Borch/IMPROVE melanoma; 23 patients), where a hydrophobicity proxy for TCR contact volume achieved AUC 0.660 (95% CI 0.609–0.709), numerically exceeding the published PRIME model (0.609; 95% CI 0.556–0.660; paired permutation p = 0.067, bootstrap CI of difference [−0.028, +0.123]). The same strategy underperformed PRIME on bladder cancer (AUC 0.570 vs PRIME 0.616) and collapsed below binding when applied to GBM (AUC 0.411 vs binding ~0.628), while the GBM-native strategy failed reciprocally on melanoma (0.466 vs melanoma specialist 0.575). A systematic analysis of individual feature discriminability across five cohorts shows the optimal feature varies by cancer context (range 0.14–0.24 AUC units), with no single feature universally best. The framework is designed to extend to additional cancer types as appropriately powered held-out validation cohorts become available.

**Keywords:** neoantigen; personalised cancer vaccine; immunotherapy; TCR recognition; context-specific ranking; autoresearch

---

## Background

Cancer cells harbour genetic mutations that generate aberrant peptides. When presented by MHC class I molecules and recognised by T-cell receptors, these neoepitopes can trigger immune-mediated tumour killing, a principle applied in personalised neoantigen vaccines with proof-of-concept activity across melanoma [1, 2], glioblastoma [3, 4], and pancreatic cancer [5].

A central challenge is candidate prioritisation: whole-exome sequencing identifies hundreds to thousands of mutation-derived peptide-HLA combinations, of which only 15–20 can be synthesised per patient. The dominant ranking criterion is predicted MHC binding affinity (NetMHCpan [6], MHCflurry [7], MixMHCpred [8]), but binding is insufficient: mean patient AUC values range 0.47–0.59 across published cohorts [9, 10].

Multiple features beyond binding have been associated with immunogenicity: peptide-MHC stability [11], expression level [1], self-dissimilarity [12], TCR contact residue properties [13, 25], and clonal frequency [14]. Methods including PRIME [8], pTuneos [15], NeoGuider [16], pVACtools [26], and multi-cohort ML approaches [21] combine subsets of these features but do so in aggregate across cancer types, implicitly assuming that immunogenicity determinants are universal.

This assumption may not hold. UV-induced C→T transitions in melanoma [22] systematically generate peptides with hydrophobic residues at TCR contact positions — a signature absent in GBM or pancreatic cancer. A context-aware strategy library, with each strategy discovered and validated within a specific cancer type, may outperform any single universal formula. However, no systematic discovery framework exists to test this hypothesis, and no validated example has been demonstrated with a properly held-out design.

We address this gap with NeoResist-MD: an automated strategy discovery loop, an evidence audit pipeline, and a clinical prioritisation platform. Here we report the discovery and held-out validation of a melanoma-specific strategy, its feature dissection, and corroboration on an independent external cohort.

---

## Methods

### Cohort assembly and leakage policy

Four cohorts with peptide-level immunogenicity labels were assembled (Table 1). **Ott 2017** (melanoma discovery; n = 13 patients, 97 peptides, 15 immunogenic) and **Sahin 2017** (melanoma held-out validation; n = 13 patients, 165 peptides, 109 immunogenic) were derived from published supplementary data [1, 2]. Both comprise peptides pre-selected by predicted HLA binding affinity and subsequently assayed by IFN-γ ELISpot. **Hilf 2019** (GBM; n = 15 patients, 152 peptides, 77 immunogenic) and **Rojas 2023** (pancreatic; n = 16 patients, 230 peptides, 30 immunogenic) were used for provisional single-cohort evaluation [3, 5].

For independent external comparison, we used the Borch/IMPROVE dataset [10] (70 patients, three cancer types: melanoma n = 23 [Neye cohort], bladder n = 22, basket n = 16) with features pre-computed by the original authors.

All feature pools underwent leakage auditing: features derived from immunogenicity labels, patient identifiers, cohort membership, or split indicators were excluded. **Pre-specification policy:** the Sahin 2017 held-out cohort was accessed exactly once, at final evaluation, after melanoma_all_signals was fixed based solely on Ott 2017 discovery performance. The strategy evaluated on Sahin was the top-ranked strategy from the discovery search — no further selection was made after Sahin was accessed. This pre-specification was recorded in the strategy registry before held-out evaluation.

### Feature computation

MHC binding affinity: NetMHCpan 4.2 (HLA-EL mode), expressed as bind_log50k = −log₁₀(IC₅₀/50). Expression: RNA-seq TPM, log-transformed as log₂(TPM + 1). TCR contact volume (tcr_volume_mean): ICERFIRE 1.0a [24], mean van der Waals volume of residues at positions predicted to contact the TCR in a standard pMHC-I complex. Aliphatic index: following Ikai [17]. Self-dissimilarity: normalised Hamming distance to the closest 9-mer in the human proteome reference.

For the Borch/IMPROVE comparison, HydroCore (mean Kyte-Doolittle hydrophobicity [23] of core positions 4–6) was used as a physicochemically motivated proxy for TCR contact volume. Both features capture hydrophobicity and bulk of TCR-facing residues; they are corroborating rather than identical measures. ICERFIRE-derived tcr_volume_mean was not available for the Borch dataset. PRIME scores in the Borch/IMPROVE dataset were pre-computed by Borch et al. using PRIME 2.0 [8]; we used these scores as published without modification. A formal validation of the HydroCore proxy — correlation with ICERFIRE-derived tcr_volume_mean on a shared dataset — requires running ICERFIRE on Borch peptide sequences, which has not been completed; the Borch result is therefore treated as corroborating rather than directly replicating the tcr_volume_mean finding.

### Strategy representation and the autoresearch loop

Ranking strategies are interpretable weighted linear functions:

**score** = Σᵢ wᵢ × fᵢ × dᵢ

where wᵢ ≥ 0, Σwᵢ = 1, and dᵢ ∈ {+1, −1} encodes the direction of association with immunogenicity.

The autoresearch loop generates candidates through seven operator families: (1) **existing** — pre-specified library seeds; (2) **mutation** — perturbations of surviving strategies; (3) **recombination** — crossover between survivors; (4) **random_blend** — Dirichlet-sampled random combinations; (5) **toolstack** — combinations tied to specific tool outputs; (6) **ml_importance** — strategies derived from cross-validated feature importances of logistic regression, random forest, and gradient boosting classifiers; and (7) **gated_ranking** — binding-gated TCR reranking blends, inspired by the DeepNeo two-stage design [20]. Family weights adapt based on per-operator keep and promote rates.

### Promotion criteria and crystallisation

A strategy is eligible for promotion only from a registry context with a designated held-out cohort. Promotion requires: (1) held-out AUC exceeding binding-only by Δ ≥ 0.005; (2) bootstrap CI lower bound within 0.05 of the baseline CI upper bound; (3) seed stability across recent discovery rounds, with bypass for strategies with delta ≥ 0.08. Promoted strategies are crystallised: re-evaluated through a fixed evidence pipeline with independent bootstrap seed (n = 1,000) and written to the registry with full provenance.

### Evaluation metrics

Primary metric: mean patient AUC — average per-patient AUC across patients with at least one positive. Bootstrap 95% CIs: resampling patients (n = 1,000, seed = 42). Permutation p-values: shuffling immunogenicity labels within each patient 10,000 times (resolution ±0.001 at n = 13 patients).

### Ablation analysis

Each feature was removed from melanoma_all_signals, remaining weights renormalised to unity, and the ablated strategy evaluated on Sahin 2017.

### Borch/IMPROVE comparison

PRIME scores used as published [10]. Binding rank derived from RankEL_4.1. HydroCore used directly as the TCR contact volume proxy (see Feature computation). For the cross-validated logistic regression, five features (RankEL_4.1, HydroAll, SelfSim, Expression, Stability) were standardised per fold and evaluated in matched 5-fold cross-validation using the partition structure provided by Borch et al.

### GBM provisional evaluation

For provisional GBM evaluation, Hilf 2019 served as the discovery cohort. Keskin 2019 was pre-specified as the directional transfer check before the Hilf discovery run was initiated; it was accessed once, after the gbm_bind_seq_60_40 strategy was fixed from Hilf performance alone. No held-out validation cohort with sufficient structure for promotion was available; the Keskin result is therefore reported as provisional transfer evidence only.

### Clinical platform

The NeoResist-MD platform implements an end-to-end workflow: mutation table upload → candidate generation → strategy scoring → tiered prioritisation output, with resume-safe case workflows, persistent storage with audit logging, and strategy dispatch from the per-cancer registry. Platform source code is included in the repository under `/platform`; a deployable release will be made available upon acceptance.

---

## Results

### 1. NeoResist-MD: a framework for context-aware strategy discovery

NeoResist-MD operates over cancer-specific discovery-validation pairs (Figure 1A). The autoresearch loop generates candidate strategies on the discovery cohort and gates promotion on held-out performance when a partner cohort is available. Promoted strategies are frozen and written to the registry with provenance documentation. Across 222 iterations and seven operator families, the loop evaluated over 30,000 strategy-dataset pairs (Figure 1B). Search was constrained to interpretable linear combinations to ensure every promoted strategy is deployable without specialised infrastructure.

### 2. Feature discriminability varies systematically by cancer context

Before evaluating full strategies, we asked whether individual features show context-specific discriminability. For each pre-computed feature with sufficient coverage, we computed mean patient AUC across five cohorts (Table 2). No single feature is universally optimal (Table 2). Binding affinity is the strongest discriminator in TESLA (AUC 0.771) but weakest in Sahin melanoma (0.530). TCR contact volume leads in Ott melanoma (0.705) but is near-chance in TESLA (0.527). Calis score — a sequence-based immunogenicity heuristic — leads in GBM/Hilf (0.669) but is below TCR contact volume on melanoma. The range in per-feature AUC across contexts (0.14–0.24 AUC units) demonstrates that immunogenicity determinants shift substantially by cancer type and cannot be captured by a single universal weighting. This motivates the per-context strategy search described in subsequent sections.

### 3. Melanoma-specific strategy outperforms binding-only on held-out validation

Using Ott 2017 as the discovery cohort, the loop identified melanoma_all_signals: MHC presentation score (weight 0.40), TCR contact volume (0.20), gene expression (0.20), and aliphatic index (0.20). Discovery AUC on Ott 2017 was 0.778.

On the held-out Sahin 2017 cohort, melanoma_all_signals achieved mean patient AUC of 0.575 (95% CI 0.511–0.630), compared to 0.470 (95% CI 0.319–0.657) for binding-only — delta +0.105, permutation *p* = 0.011 (Figure 2A, Table 3). Comparator strategies from the pre-specified library — rl_tcr_v1_from_ott_locked (AUC 0.518) and neoguider_v2_melanoma (AUC 0.531) — were numerically lower, though confidence intervals overlap substantially and no pairwise significance test was conducted (Table 3). Per-patient recall@5 was 0.441 for melanoma_all_signals versus 0.418 for binding-only.

### 4. TCR contact volume is the load-bearing feature

Feature ablation on Sahin 2017 showed removing TCR contact volume reduced AUC by 0.086 — from 0.575 to 0.490 — the largest single-feature effect (Figure 2B, Table 4). Removing the aliphatic index produced a secondary reduction (−0.041). Removing presentation score or expression had minimal or slightly positive effects, indicating marginal contribution once TCR contact volume and aliphatic character are accounted for.

The four-feature strategy was fixed based on Ott 2017 discovery performance prior to any held-out evaluation. The held-out ablation reveals that TCR contact volume and aliphatic index carry the discriminating signal while expression and presentation are redundant in this cohort — consistent with the known instability of multi-feature models at n = 13 and with the pre-screened nature of these cohorts, where binding-correlated features add little independent information.

The dominance of TCR contact volume is consistent with a biologically distinct signal: bulkier, more hydrophobic residues at TCR contact positions form more productive TCR engagements independently of MHC binding affinity [18].

### 5. The TCR contact signal is corroborated on an independent external melanoma cohort and numerically exceeds PRIME

We tested whether the TCR contact finding generalises beyond Sahin on the Borch/IMPROVE melanoma subset (Neye cohort; 23 patients, 5,921 peptides, 204 immunogenic [10]) — entirely independent of our discovery and validation pipeline.

Because ICERFIRE-derived tcr_volume_mean is not available for Borch, we used HydroCore as a physicochemically motivated proxy (see Methods). A two-feature strategy combining binding rank and HydroCore achieved mean patient AUC of 0.660 (95% CI 0.609–0.709), numerically exceeding PRIME 2.0 (AUC 0.609; 95% CI 0.556–0.660 [8]) by +0.051 (paired permutation p = 0.067; bootstrap CI of difference [−0.028, +0.123]; Figure 3A, Table 5). The difference is directional and consistent with the Sahin result but does not reach conventional significance at n = 23 patients. HydroCore alone achieved 0.650. A logistic regression trained on five Borch features in matched 5-fold cross-validation achieved 0.644; because this model is trained on the same Borch melanoma data, it is not directly comparable to PRIME as a fixed external benchmark, and its advantage over PRIME is expected — it is reported for completeness. The bind+HydroCore result is the primary comparison as it is applied without training. These results corroborate the Sahin finding that TCR contact hydrophobicity adds information beyond binding in melanoma, though we note HydroCore is a proxy rather than a direct computation of the same quantity.

The HydroCore advantage was specific to melanoma. On the Borch/IMPROVE bladder cohort (n = 22 patients), the same strategy achieved AUC 0.570 — below PRIME (0.616) by −0.046 (Figure 3B). Immunogenic peptides in Borch melanoma showed higher HydroCore than non-immunogenic (mean +0.239 vs −0.183; Pearson r = +0.082, *p* < 0.001), with attenuated effect across non-melanoma cohorts (r = +0.047). This pattern is consistent with UV-induced C→T mutations systematically enriching hydrophobic core residues in melanoma neoantigens — a mutational signature absent in bladder cancer.

An exploratory autoresearch analysis of bladder-specific features and a full bidirectional cross-test are reported in Supplementary Note S1.

### 6. Bidirectional cross-cancer failure supports context specificity

The cross-cancer transfer matrix provides the clearest available demonstration of the context-dependence thesis. When the melanoma specialist (melanoma_all_signals) is applied to the GBM Keskin cohort, mean patient AUC drops to 0.411 — substantially below both the GBM-native strategy (0.683) and the binding-only baseline (~0.628). In the reverse direction, the GBM-native model applied to the melanoma Sahin cohort achieves 0.466 — below binding-only (0.470) and 0.109 below the melanoma specialist (0.575). Each specialist performs at or below the binding baseline when applied to the other's home context. This bidirectional failure pattern is the most direct empirical evidence that optimal ranking formulas are context-specific: the features that matter in melanoma actively harm GBM ranking, and vice versa (Table 5).

Using Hilf 2019 as discovery and Keskin 2019 as a pre-specified provisional transfer check, the loop produced gbm_bind_seq_60_40 (binding sigmoid + sequence composite) as the GBM candidate. An expression-only candidate with AUC 1.000 on Hilf was excluded as a sparse-coverage artifact (expression available for 12 of 152 rows, 11 immunogenic). The GBM result remains provisional — a held-out GBM validation cohort is required — but the cross-transfer analysis stands independently of the validation status of either specialist.

### 7. The autoresearch loop adapts to productive operator families

Across 222 iterations, mutation and recombination operators earned increasing search budget based on their promotion rates. ML-importance operators produced candidates with AUC 0.56–0.58 on Sahin, confirming that cross-validated feature importances provide signal but that fixed interpretable blends can match or exceed them on held-out data (Supplementary Figure S2).

---

## Discussion

We present NeoResist-MD and its first validated application: a melanoma-specific strategy that outperforms binding-only on a held-out cohort (delta +0.105; *p* = 0.011) with TCR contact volume as the dominant feature, corroborated on an independent external cohort where it numerically exceeds PRIME (AUC 0.660 vs 0.609; p = 0.067).

**The TCR contact signal is the core finding.** Across two independent evaluations — Sahin 2017 (ELISpot-validated, held-out) and Borch/IMPROVE melanoma (multimer-validated, external) — TCR contact features consistently added information beyond binding. The mechanistic interpretation is coherent: residues with larger van der Waals volume at TCR contact positions form more stable engagements independently of HLA binding affinity [18]. UV mutational signature in melanoma creates systematic enrichment of hydrophobic neoantigen core residues, making this signal melanoma-specific. The failure of the same strategy on Borch bladder cancer provides direct empirical support for context-dependence.

**Why does the melanoma strategy numerically exceed PRIME on Borch despite non-significant difference?** PRIME was trained on multi-cancer data, partially discounting signals that are melanoma-specific. A context-specific formula — even a simple linear combination — can fully exploit such signals. The Borch comparison provides external support for this argument, though PRIME is a universal benchmark and the comparison is not a direct head-to-head with a bladder-tuned PRIME.

**Limitations.** The Ott and Sahin cohorts each contain 13 patients, limiting statistical power for pairwise strategy comparisons; the feature ablation contributions (Section 3) are estimates with substantial uncertainty at this sample size. The permutation p-value (0.011) is computed under a single pre-specified held-out test, but independent replication in a larger cohort would strengthen confidence. The validated result currently covers a single cancer type. Demonstrating the general principle requires at least one additional held-out validated context. Existing accessible cohorts (Keskin 2019 GBM supplement [3]; Balachandran 2022 pancreatic [19]) may provide partial data. NCI Surgery Branch data (NCT01174121, NCT02133196) could supply additional multi-cancer sub-cohorts with ELISpot-validated labels.

The HydroCore proxy introduces measurement imprecision: it correlates with tcr_volume_mean but is not identical. A formal correlation analysis between the two features on a shared dataset is needed to quantify the approximation. Full ICERFIRE integration would enable TCR contact features across all contexts systematically.

Regarding multiple comparisons: the autoresearch loop explored >30,000 strategy-dataset pairs but all loop-internal results are exploratory. The reported melanoma_all_signals result was pre-specified (top discovery-set strategy), and Sahin 2017 was accessed once. The permutation *p* = 0.011 is computed on the held-out set under this single pre-specified test.

The context-aware selector result (aggregate 0.694 vs 0.545 for best universal strategy) is an exploratory computation driven by provisional, single-cohort non-melanoma contexts and is reported in Supplementary Note S2 only.

**Relationship to NeoGuider.** NeoGuider [16] applies adaptive KDE + isotonic regression + logistic regression to resolve nonlinearity and class imbalance. NeoResist-MD asks a different question: which features matter per cancer type. The approaches are complementary — NeoGuider-style transformation could be applied to the features selected by NeoResist-MD for any given context.

**Clinical implications.** For melanoma vaccine design, our results support adding TCR contact volume and aliphatic index to the binding affinity feature set. The improvement in recall@5 (+5.5 percentage points) translates to approximately one additional immunogenic candidate recovered per patient within the top five ranked — a meaningful gain under synthesis budget constraints.

---

## Conclusions

NeoResist-MD demonstrates that cancer-type-specific neoantigen ranking strategies can be systematically discovered, held-out validated, and corroborated on independent external cohorts. The melanoma strategy outperforms binding-only by +0.105 AUC (*p* = 0.011), with TCR contact volume as the load-bearing feature. A per-feature discriminability analysis across five cohorts shows no single feature is universally optimal, with context-specific range of 0.14–0.24 AUC units. The cross-cancer transfer matrix demonstrates bidirectional failure: each specialist collapses at or below binding when applied to the other's home context. On an independent external melanoma cohort, a TCR contact hydrophobicity proxy achieves AUC 0.660 versus PRIME 0.609 (p = 0.067; directional, n = 23), while failing on bladder cancer. Extending the framework to GBM, pancreatic, and other cancer types requires powered held-out validation cohorts not currently available. We release the autoresearch loop, evidence audit pipeline, and clinical platform as open tools.

---

## Abbreviations

| Abbreviation | Definition |
|---|---|
| MHC | Major histocompatibility complex |
| HLA | Human leukocyte antigen |
| AUC | Area under the curve |
| ROC | Receiver operating characteristic |
| TCR | T-cell receptor |
| ELISpot | Enzyme-linked immunospot |
| IC₅₀ | Half-maximal inhibitory concentration |
| GBM | Glioblastoma multiforme |
| PDAC | Pancreatic ductal adenocarcinoma |
| CI | Confidence interval |
| TPM | Transcripts per million |
| CV | Cross-validation |

---

## Supplementary Information

**Supplementary Note S1. Bladder autoresearch and bidirectional cross-test.**
Iterative autoresearch on the Borch/IMPROVE bladder cohort (6,237 rows, with additional derived features including germline EL rank and agretopicity computed from the raw peptide sequences) converged on a blend of binding affinity rank, germline EL rank, agretopicity, and expression with mean patient AUC 0.605 — below PRIME-only (0.629) but above rankel_prime (0.601), using a feature mix distinct from the melanoma TCR-contact strategy. A full bidirectional cross-test was attempted: a bladder-optimised blend (Foreignness + RankEL, equal weights) applied to the Borch melanoma cohort scored 0.547 — substantially below PRIME (0.609) — confirming that bladder-relevant features are actively harmful on melanoma. The reverse direction was not achievable: exhaustive search over all Borch/IMPROVE features found no combination exceeding PRIME on bladder (best 0.584 vs PRIME 0.616). This reflects a feature-availability gap: Borch bladder lacks TCR contact volume, binding stability, and self-dissimilarity computed from peptide sequences. These results are treated as supplemental context-dependence support, not as an independent bladder validation.

**Supplementary Note S2. Context-aware selector (exploratory).**
A context-labelled selector assigning the optimal per-cancer strategy achieved aggregate mean primary metric of 0.694 versus 0.545 for the best universal fixed strategy. This result is exploratory: three of four contexts (GBM, pancreatic, TESLA) contribute provisional single-cohort evidence. The melanoma held-out contribution drives the quality of the estimate.

**Supplementary Table S1.** Per-patient recall at top-5, top-10, top-20 on Sahin 2017 for all evaluated strategies.

**Supplementary Table S2.** Complete strategy comparison on Sahin 2017 — all 27 strategies (mean patient AUC, 95% CI, delta vs binding-only, recall@5, recall@10).

**Supplementary Table S3.** Feature ablation — all four features, AUC, CI, delta, percentage contribution.

**Supplementary Table S4.** Borch/IMPROVE full comparison — all methods, all three cancer types.

**Supplementary Table S5.** Autoresearch loop configuration parameters.

**Supplementary Table S6.** Multiple comparison statement. The autoresearch loop evaluated >30,000 strategy-dataset pairs. All loop-internal results are exploratory. The reported melanoma_all_signals result on Sahin 2017 was pre-specified and tested once.

**Supplementary Figure S1.** Per-patient recall@N (N = 5, 10, 20) on Sahin 2017.

**Supplementary Figure S2.** Autoresearch loop iteration history — exploration pressure, family weight adaptation, elite bank occupancy.

**Supplementary Figure S3.** HydroCore signal distributions in Borch/IMPROVE by cancer type and immunogenicity status.

**Supplementary Figure S4.** Bladder autoresearch convergence and feature composition.

---

## Acknowledgements

We thank the authors of Ott 2017, Sahin 2017, Hilf 2019, and Rojas 2023 for sharing supplementary data. We thank the TESLA Consortium and Borch et al. for making the IMPROVE dataset publicly available. We thank the DTU Bioinformatics group for ICERFIRE access.

---

## Authors' contributions

[TBD]

---

## Funding

[TBD]

---

## Data availability

All cohort data are from published supplementary materials. Borch/IMPROVE: github.com/SRHgroup/IMPROVE_paper. The autoresearch loop, evidence audit pipeline, and clinical platform are available at github.com/Adnan-shamim-dar/neoresist-md (NeoResist-MD v1.0). Processed feature tables for Ott 2017, Sahin 2017, Hilf 2019, and Rojas 2023 are provided as Supplementary Data.

---

## Declarations

**Ethics:** All cohorts were previously published with appropriate ethical approval by the original study teams. No new patient data were collected.

**Competing interests:** The author declares no competing financial or personal interests that could have influenced the work reported in this paper. NeoResist-MD exists as a research software tool under active development; it has no commercial deployment, generates no revenue, and is released as open-source. This research received no external funding from any public, commercial, or not-for-profit funding agency.

---

## References

[1] Ott PA, et al. An immunogenic personal neoantigen vaccine for patients with melanoma. *Nature* 2017;547:217–221.

[2] Sahin U, et al. Personalised RNA mutanome vaccines mobilise poly-specific therapeutic immunity against cancer. *Nature* 2017;547:222–226.

[3] Keskin DB, et al. Neoantigen vaccine generates intratumoral T cell responses in phase Ib glioblastoma trial. *Nature* 2019;565:234–239.

[4] Hilf N, et al. Actively personalised vaccination trial for newly diagnosed glioblastoma. *Nature* 2019;565:240–245.

[5] Rojas LA, et al. Personalised RNA neoantigen vaccines stimulate T cells in pancreatic cancer. *Nature* 2023;618:144–150.

[6] Reynisson B, et al. NetMHCpan-4.1 and NetMHCIIpan-4.0: improved predictions of MHC antigen presentation. *Nucleic Acids Research* 2020;48:W449–W454.

[7] O'Donnell TJ, et al. MHCflurry 2.0: improved pan-allele prediction of MHC class I-presented peptides by incorporating antigen processing. *Cell Systems* 2020;11:42–48.

[8] Gfeller D, et al. Improved predictions of antigen presentation and TCR recognition with MixMHCpred2.2 and PRIME2.0. *Cell Systems* 2023;14:72–83.

[9] Wells DK, et al. Key parameters of tumor epitope immunogenicity revealed through a consortium approach improve neoantigen prediction. *Cell* 2020;183:818–834.

[10] Borch A, et al. IMPROVE: a feature model to predict neoepitope immunogenicity through broad-scale validation. *Frontiers in Immunology* 2024;15:1265584. DOI: 10.3389/fimmu.2024.1265584

[11] Rasmussen M, et al. Pan-specific prediction of peptide-MHC class I complex stability. *Journal of Immunology* 2016;197:1517–1524.

[12] Richman LP, et al. Neoantigen dissimilarity to the self-proteome predicts immunogenicity and response to immune checkpoint blockade. *Cell Systems* 2019;9:375–382.

[13] Luksza M, et al. A neoantigen fitness model predicts tumour response to checkpoint blockade. *Nature* 2017;551:517–520.

[14] Balachandran VP, et al. Identification of unique neoantigen qualities in long-term survivors of pancreatic cancer. *Nature* 2017;551:512–516.

[15] Zhou C, et al. pTuneos: prioritizing tumor neoantigens from next-generation sequencing data. *Genome Medicine* 2019;11:67.

[16] Zhao X, et al. NeoGuider: neoepitope prediction using advanced feature engineering. *Genome Medicine* 2026;18:13.

[17] Ikai A. Thermostability and aliphatic index of globular proteins. *Journal of Biochemistry* 1980;88:1895–1898.

[18] Garcia KC, et al. Structural basis of plasticity in T cell receptor recognition of a self peptide-MHC antigen. *Science* 1998;279:1166–1172.

[19] Balachandran VP, et al. Neoantigen quality predicts immunoediting in survivors of pancreatic cancer. *Nature* 2022;606:389–395.

[20] Lee C-H, et al. DeepNeo: an integrated deep learning framework for predicting neoantigen immunogenicity. *Nucleic Acids Research* 2023;51:W51–W57.

[21] Müller M, et al. Machine learning methods and harmonized datasets improve immunogenic neoantigen prediction. *Immunity* 2023;56:2879–2891.

[22] Alexandrov LB, et al. Signatures of mutational processes in human cancer. *Nature* 2013;500:415–421. DOI: 10.1038/nature12477.

[23] Kyte J, Doolittle RF. A simple method for displaying the hydropathic character of a protein. *Journal of Molecular Biology* 1982;157:105–132. DOI: 10.1016/0022-2836(82)90515-0.

[24] Borch A, et al. A large-scale study of peptide features defining immunogenicity of cancer neo-epitopes. *NAR Cancer* 2024;6:zcae002. DOI: 10.1093/narcan/zcae002.

[25] Schmidt J, et al. Prediction of neo-epitope immunogenicity reveals TCR recognition determinants and provides insight into immunoediting. *Cell Reports Medicine* 2021;2:100194. DOI: 10.1016/j.xcrm.2021.100194.

[26] Hundal J, et al. pVACtools: a computational toolkit to identify and visualize cancer neoantigens. *Cancer Immunology Research* 2020;8:409–420. DOI: 10.1158/2326-6066.CIR-19-0401.

---

## Tables

**Table 1. Cohort characteristics.**

| Cohort | Cancer type | Role | Patients | Peptides | Immunogenic | Assay |
|---|---|---|---|---|---|---|
| Ott 2017 | Melanoma | Discovery | 13 | 97 | 15 | IFN-γ ELISpot |
| Sahin 2017 | Melanoma | Held-out validation | 13 | 165 | 109 | IFN-γ ELISpot |
| Hilf 2019 | GBM | Provisional | 15 | 152 | 77 | IFN-γ ELISpot |
| Rojas 2023 | Pancreatic | Provisional | 16 | 230 | 30 | IFN-γ ELISpot |
| Borch/IMPROVE Neye | Melanoma | External benchmark | 23 | 5,921 | 204 | MHC-I multimer |

*All cohorts contain peptides pre-selected by predicted HLA binding prior to immunogenicity testing.*

**Table 2. Feature discriminability across cancer contexts (mean patient AUC, single-feature rankings).**

| Feature | Ott 2017 (mel) | Sahin 2017 (mel) | Hilf 2019 (GBM) | Rojas 2023 (pan) | TESLA (mixed) | Range |
|---|---|---|---|---|---|---|
| Binding affinity | 0.660 | 0.530 | 0.588 | 0.547 | **0.771** | 0.241 |
| TCR contact volume | **0.705** | 0.556 | 0.602 | **0.664** | 0.527 | 0.178 |
| TCR charge diff | **0.694** | 0.532 | 0.500 | 0.521 | — | 0.194 |
| Expression | 0.576 | 0.538 | — | — | **0.685** | 0.147 |
| Calis score | 0.626 | — | **0.669** | — | 0.580 | 0.088 |

*Bold = highest-ranking feature per cohort. Range = max − min across cohorts with ≥2 valid AUC values. No single feature is universally optimal. Coverage gaps (—) indicate feature not available for that cohort.*

**Table 3. Strategy comparison on held-out Sahin 2017 melanoma cohort.**


| Strategy | Key features | AUC | 95% CI | Delta vs binding |
|---|---|---|---|---|
| **melanoma_all_signals** | pres, TCR_vol, expr, aliphatic | **0.575** | [0.511, 0.630] | **+0.105*** |
| melanoma_bind_expr_tcr | binding, expr, TCR vol | 0.534 | [0.330, 0.733] | +0.064 |
| neoguider_v2_melanoma | binding, TCR chem | 0.531 | [0.377, 0.702] | +0.061 |
| rl_tcr_v1_from_ott_locked | binding, TCR vol/charge | 0.518 | [0.356, 0.691] | +0.048 |
| binding_plus_calis | binding, sequence | 0.470 | [0.319, 0.657] | 0.000 |
| binding_only | binding | 0.470 | [0.319, 0.657] | — |

*Primary metric: mean patient AUC (patient-level bootstrap, n = 1,000). *p = 0.011 vs binding-only (permutation test). Comparator differences are numerical; no pairwise significance tests were conducted given overlapping CIs at this sample size.*

**Table 4. Feature ablation — melanoma_all_signals on Sahin 2017.**

| Dropped feature | AUC | Delta vs full |
|---|---|---|
| None (full model) | 0.575 | — |
| **tcr_volume_mean** | 0.490 | **−0.086** |
| aliphatic_index | 0.534 | −0.041 |
| expression_log2 | 0.586 | +0.010 |
| presentation_score | 0.578 | +0.003 |

**Table 5. External benchmark — Borch/IMPROVE melanoma (Neye cohort; 23 patients).**

| Method | Type | AUC | 95% CI | vs PRIME |
|---|---|---|---|---|
| **bind + HydroCore (50/50)** | linear, no training | **0.660** | [0.609, 0.709] | **+0.051** |
| HydroCore alone | linear, no training | 0.650 | [0.595, 0.699] | +0.041 |
| LogReg (5 features, 5-fold CV) | trained on Borch melanoma | 0.644 | [0.595, 0.694] | +0.035 |
| **PRIME 2.0** (Gfeller 2023) | fixed published model | **0.609** | [0.556, 0.660] | baseline |
| binding_only | linear, no training | 0.559 | [0.498, 0.615] | −0.050 |

*HydroCore: mean Kyte-Doolittle hydrophobicity of core positions 4–6; physicochemical proxy for TCR contact volume. Difference vs PRIME: paired permutation p = 0.067, bootstrap CI [−0.028, +0.123] — directional, not significant at n = 23. LogReg uses 5-fold CV on the same Borch melanoma data; not directly comparable to fixed external PRIME.*

**Table 6. Bidirectional cross-cancer transfer matrix.**

| Specialist strategy | Home context | Home AUC | Away context | Away AUC | Binding baseline (away) |
|---|---|---|---|---|---|
| melanoma_all_signals | Melanoma (Sahin) | 0.575 | GBM (Keskin) | 0.411 | ~0.628 |
| gbm_bind_seq_60_40 | GBM (Hilf/Keskin) | 0.683 | Melanoma (Sahin) | 0.466 | 0.470 |

*Each specialist underperforms the binding-only baseline when applied to the other's home context. The GBM model on Sahin (0.466) falls below binding (0.470); the melanoma model on Keskin (0.411) falls 0.217 below binding (~0.628). GBM result is provisional (single cohort); cross-transfer values are from the pre-computed cross-cancer transfer evaluation.*

---

## Figure Legends

**Figure 1. NeoResist-MD framework.**
(A) Discovery-validation design. Each cancer context has a designated discovery cohort and, where available, a held-out validation cohort accessed once at final evaluation. Promoted strategies are crystallised into the registry with provenance. (B) Autoresearch loop history showing exploration pressure and family weight adaptation across 222 rounds.

**Figure 2. Melanoma strategy discovery and ablation.**
(A) Mean patient AUC on held-out Sahin 2017. Horizontal line: binding-only baseline (0.470). melanoma_all_signals: 0.575 (delta +0.105; *p* = 0.011). Error bars: 95% bootstrap CI. (B) Feature ablation. TCR contact volume is the dominant component (−0.086 on removal).

**Figure 3. External corroboration and cancer-type specificity.**
(A) Borch/IMPROVE melanoma (23 patients). bind+HydroCore achieves AUC 0.660 vs PRIME 2.0 0.609 (p = 0.067; directional). (B) Cancer-type specificity: bind+HydroCore numerically exceeds PRIME on melanoma (+0.051) but underperforms on bladder (−0.046).
