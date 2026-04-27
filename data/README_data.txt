NeoResist-MD — Dataset Sources and Attribution
===============================================

All datasets used in this study are derived from published supplementary
materials. No new patient data were collected.

------------------------------------------------------------------------
DATASETS
------------------------------------------------------------------------

1. Ott 2017 (melanoma — DISCOVERY cohort)
   Source: Ott PA, et al. An immunogenic personal neoantigen vaccine for
           patients with melanoma. Nature 2017;547:217-221.
   DOI: 10.1038/nature22991
   Usage: Strategy discovery (n=13 patients, 97 peptides, 15 immunogenic)

2. Sahin 2017 (melanoma — HELD-OUT VALIDATION cohort)
   Source: Sahin U, et al. Personalised RNA mutanome vaccines mobilise
           poly-specific therapeutic immunity against cancer.
           Nature 2017;547:222-226.
   DOI: 10.1038/nature23003
   Usage: Single-access held-out evaluation (n=13 patients, 165 peptides,
          109 immunogenic). Accessed once after strategy was fixed.

3. Hilf 2019 (GBM — provisional)
   Source: Hilf N, et al. Actively personalised vaccination trial for
           newly diagnosed glioblastoma. Nature 2019;565:240-245.
   DOI: 10.1038/s41586-018-0810-y
   Usage: GBM discovery (n=15 patients, 152 peptides, 77 immunogenic)

4. Rojas 2023 (pancreatic — provisional)
   Source: Rojas LA, et al. Personalised RNA neoantigen vaccines stimulate
           T cells in pancreatic cancer. Nature 2023;618:144-150.
   DOI: 10.1038/s41586-023-06063-y
   Usage: Pancreatic discovery (n=16 patients, 230 peptides, 30 immunogenic)

5. Borch/IMPROVE 2024 (external benchmark — NOT used for discovery/training)
   Source: Borch A, et al. IMPROVE: a feature model to predict neoepitope
           immunogenicity through broad-scale validation.
           Frontiers in Immunology 2024;15:1265584.
   DOI: 10.3389/fimmu.2024.1265584
   Repository: github.com/SRHgroup/IMPROVE_paper
   Usage: External benchmark only (70 patients, 3 cancer types)

------------------------------------------------------------------------
LICENSE AND ETHICS
------------------------------------------------------------------------

All cohorts were previously published with appropriate ethical approval
by the original study teams. Users of this repository must comply with
the terms of the original data publications. Do not use for commercial
purposes without explicit permission from the original authors.

------------------------------------------------------------------------
FEATURE COMPUTATION
------------------------------------------------------------------------

Binding affinity: NetMHCpan 4.2 (HLA-EL mode)
TCR contact volume: ICERFIRE 1.0a (DTU Bioinformatics)
Expression: RNA-seq TPM from original supplementary data
Self-dissimilarity: normalised Hamming distance, human proteome reference
Aliphatic index: Ikai (1980) J Biochem 88:1895-1898
