# Cover Letter — NeoResist-MD

**To the Editors,**
**[Journal Name]**

We submit for your consideration a manuscript entitled:

**"NeoResist-MD: discovery and validation of a melanoma-specific neoantigen ranking strategy and a framework for cancer-type extension"**

---

## Why This Matters

Personalised cancer vaccines require accurate prioritisation of neoantigen candidates for synthesis. Current computational approaches apply universal scoring formulas — dominated by MHC binding affinity — uniformly across cancer types, despite growing evidence that immunogenicity determinants differ by tumour context. This universality assumption has not been rigorously tested, and no systematic tool exists for discovering and validating cancer-type-specific ranking strategies.

**Our framework resolves this gap.** NeoResist-MD demonstrates, for the first time with a properly held-out validation design, that a melanoma-specific strategy systematically outperforms universal ranking — and identifies *why*: TCR contact volume at core peptide positions is the dominant melanoma-specific discriminating feature, consistent with UV mutational signature enriching hydrophobic residues at TCR contact positions in melanoma neoantigens.

---

## Key Contributions

1. **Held-out validated melanoma result.** A strategy discovered on Ott 2017 improves mean patient AUC from 0.470 to 0.575 on the independent held-out Sahin 2017 cohort (delta +0.105; 95% CI 0.511–0.630; permutation p = 0.011). The strategy was pre-specified and Sahin was accessed exactly once.

2. **TCR contact volume as a novel melanoma-specific signal.** Feature ablation identifies TCR contact volume (ICERFIRE-derived) as the dominant discriminating component (−0.086 AUC on removal). This is independently corroborated on the Borch/IMPROVE external melanoma cohort, where a hydrophobicity proxy for TCR contact volume numerically exceeds PRIME 2.0 (AUC 0.660 vs 0.609; p = 0.067).

3. **Bidirectional context-specificity demonstrated.** The melanoma specialist collapses below the binding baseline when applied to GBM (AUC 0.411 vs binding ~0.628), and the GBM specialist reciprocally fails on melanoma (AUC 0.466 vs binding 0.470). A per-feature discriminability analysis across five independent cohorts confirms that no single feature is universally optimal (range 0.14–0.24 AUC units across contexts).

4. **An open, reproducible discovery framework.** The autoresearch loop, evidence audit pipeline, and clinical platform are released as open tools. The framework is generic: any cancer type with a discovery cohort, a held-out validation partner, and pre-computed feature tables can be added with no pipeline modifications.

---

## Fit for This Journal

This work addresses a clinically urgent problem in personalised oncology with rigorous methodology, honest reporting of effect sizes and their uncertainty, and a concrete path to multi-cancer extension. The combination of a validated biological finding, a novel framework, and transparent limitations is appropriate for [Genome Medicine / NPJ Precision Oncology].

We confirm that this manuscript has not been submitted elsewhere and that all authors have approved the submission.

Sincerely,

[Author name]
[Affiliation]
[Contact]

---

*Note: Repository URL TBD; will be included in the final submitted version.*
