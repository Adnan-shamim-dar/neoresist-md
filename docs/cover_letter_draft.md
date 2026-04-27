# Cover Letter — NeoResist-MD

**To the Editors,**
**Genome Medicine**

I am submitting for your consideration a manuscript entitled:

**"NeoResist-MD: discovery and validation of a melanoma-specific neoantigen ranking strategy and a framework for cancer-type extension"**

---

## A Personal Note

I want to be upfront about the nature of this submission. I am a single researcher — a medical student at the Università Politecnica delle Marche in Ancona — working independently, without a laboratory, without external funding, and without access to institutional compute clusters or proprietary immunogenicity datasets. What I do have is a genuine belief that the way we currently rank neoantigen candidates for personalised cancer vaccines is leaving signal on the table, and that the specific signal we are missing is different for every cancer type.

This work is the result of building everything from scratch: the discovery framework, the evaluation pipeline, the evidence audit protocol, and the clinical platform — all designed around the constraint of using only what is publicly available. The honest limitation of this paper is that I could only validate one cancer context. Not because the framework cannot do more — it is built explicitly to scale — but because the held-out cohort data required to validate a second context (GBM, pancreatic, bladder) simply does not exist in any publicly accessible form. I reached the boundary of what a single person with public data can rigorously demonstrate, and I have tried to be transparent about exactly where that boundary is and what lies beyond it.

I am submitting this now because the melanoma result is real, independently validated, and externally corroborated — and because I believe the framework itself deserves to be in the community's hands.

---

## Why This Matters

Personalised cancer vaccines require accurate prioritisation of neoantigen candidates for synthesis — typically 15 to 20 peptides per patient. The dominant criterion remains predicted MHC binding affinity, despite consistent evidence that binding alone achieves mean patient AUC of only 0.47–0.59 across published cohorts. The implicit assumption of universality — that the same scoring formula works for melanoma, GBM, and pancreatic cancer alike — has never been rigorously tested with a proper held-out validation design.

NeoResist-MD is my attempt to test it. The answer is that it does not hold: the features that discriminate immunogenicity in melanoma are actively harmful when applied to GBM, and vice versa. The mechanism is interpretable — UV-driven C→T mutations in melanoma systematically enrich hydrophobic residues at TCR contact positions, creating a signal that simply does not exist in other tumour types. Finding and validating this signal required building a systematic discovery framework, not just trying features one at a time.

---

## Key Contributions

1. **Held-out validated melanoma result.** A strategy discovered on Ott 2017 improves mean patient AUC from 0.470 to 0.575 on the independent held-out Sahin 2017 cohort (delta +0.105; 95% CI 0.511–0.630; permutation p = 0.011). The strategy was pre-specified before Sahin was accessed — documented in the repository's provenance log.

2. **TCR contact volume as the dominant melanoma-specific signal.** Feature ablation identifies TCR contact volume as the load-bearing component (−0.086 AUC on removal), corroborated on the independent Borch/IMPROVE melanoma cohort where a hydrophobicity proxy for TCR contact volume numerically exceeds PRIME 2.0 (AUC 0.660 vs 0.609; p = 0.067; directional at n = 23 patients).

3. **Bidirectional context-specificity demonstrated from existing data.** The melanoma specialist collapses below the binding baseline on GBM (0.411 vs ~0.628), and the GBM specialist fails reciprocally on melanoma (0.466 vs binding 0.470). A per-feature analysis across five cohorts shows the optimal discriminating feature varies by 0.14–0.24 AUC units across cancer contexts — no single feature is universally best.

4. **An open framework designed to scale.** The autoresearch loop runs over interpretable linear strategy combinations with adaptive operator weighting, full provenance logging, and a crystallisation protocol that freezes strategies before held-out access. Any cancer type with a discovery cohort, a held-out partner, and pre-computed feature tables can be added without modifying the pipeline. The bottleneck is data, not code.

---

## The Vision

The paper this framework is designed to eventually produce — a multi-cancer strategy registry where each cancer type has its own validated ranking formula — requires cohort data that no single researcher currently holds. It will require collaboration, data sharing, and investment in systematic immunogenicity assaying across tumour types. I cannot produce that paper alone. What I can do is build the infrastructure, demonstrate that the approach works on the one context where I have the data to prove it, and release everything openly so that others can extend it.

If this work reaches a laboratory with access to a second GBM cohort, a second pancreatic cohort, or the NCI Surgery Branch dataset, the framework is ready. The validation protocol is documented. The code runs. The only missing ingredient is the data.

I release this not as a finished multi-cancer story, but as the first chapter of one — and as an honest demonstration that a single motivated person, with public data and a clear methodology, can move the question forward.

---

## Fit for Genome Medicine

This work addresses a clinically urgent problem with rigorous methodology, honest reporting of effect sizes and their uncertainty, transparent treatment of limitations, and a concrete and reproducible framework for the community. The single-author nature of this submission is unusual; I believe the science justifies it, and I welcome the scrutiny of peer review.

I confirm this manuscript has not been submitted elsewhere and that I, as the sole author, approve the submission.

Sincerely,

**Adnan Shamim Dar**
School of Medicine, Università Politecnica delle Marche (UNIVPM), Ancona, Italy
s1119248@studenti.univpm.it

*Repository: github.com/Adnan-shamim-dar/neoresist-md*
