# ROADMAP

## Now

- Keep architecture and module contracts stable in `docs/ARCHITECTURE.md`.
- Establish deterministic module boundaries (M0-M5).
- Maintain canonical product spec in `docs/NEORESIST-MD_MASTER_PLAN.md`.
- Add schema definitions for:
  - `qc_report`
  - `neo_candidates`
  - `clones`
  - `resistance_profile`
  - `trial_matches`
- Keep AutoResearch constrained to evidence synthesis and rationale generation only.

## Next

- Extract NeoVax logic into `backend/core/neoantigen/` as pure services.
- Add CLI spine (`backend/api/cli`) to run one case end-to-end with placeholders.
- Implement M0 QC gating with schema + data sufficiency checks.
- Add placeholder but schema-valid outputs for clonality/resistance/trialmatch.
- Generate first unified markdown case report from all module outputs.

## Later

- Replace placeholder clonality with validated inference logic.
- Integrate curated resistance sources and robust trial ranking.
- Add retrospective validation datasets and confidence/audit reporting.
- Build thin web app/API on top of tested core services.
- Add FHIR-aligned export path and partner-facing integration endpoints.
