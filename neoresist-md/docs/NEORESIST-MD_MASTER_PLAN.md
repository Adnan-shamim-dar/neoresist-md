# NEORESIST-MD_MASTER_PLAN.md

## Problem

Oncologists lack an end-to-end tool to go from tumor sequencing to neoantigen and resistance interpretation plus trial matching, especially for heterogeneous tumors with multiple subclones. Existing tools are mostly research-only and not built for clinical workflow, while adoption is blocked by workflow integration gaps, fragmented data, and unresolved validation concerns.

## Vision

Drag-and-drop VCF input (optionally RNA and slide data) should automatically infer clones, predict neoantigens, identify resistance drivers, match trials, and produce both a signed clinical-style report and machine-readable JSON. The platform should plug into pharma/CRO and clinical EHR workflows for referral operations and real-world outcomes tracking.

## Modules

- **M1: NeoVax** - VCF to filtered variants and neoantigen candidates (current module).
- **M2: CloneEscape** - VAF clustering to clonal architecture and subclone neoantigens.
- **M3: ResistanceLoop** - Gene-level interpretation connected to DepMap/CRISPR resistance signals.
- **M4: TrialMatch** - Targets plus indication to NCT trial matching with eligibility scoring.
- **M5: Report & API** - Human-readable PDF output plus machine-readable JSON output.

## Non-Negotiables

- End-to-end flow from one upload.
- Every output tied to citations (papers and trials) so results are defensible to MDs and regulators.
- Designed from day one so pharma/CRO partners can pay for trial referral plus data operations (Medvi-style "own the pipe" model rather than owning the molecule).
