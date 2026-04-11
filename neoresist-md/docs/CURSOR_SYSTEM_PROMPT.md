# CURSOR_SYSTEM_PROMPT

## Product Identity

NeoResist-MD is an end-to-end oncology workflow platform from tumor data to clonal neoantigens, resistance insight, and trial matching.
It is not just a neoantigen predictor.

## Architecture Summary

- Deterministic core modules in `backend/core/`:
  - M0 QC
  - M1 NeoVax
  - M2 CloneEscape
  - M3 ResistanceLoop
  - M4 TrialMatch
  - M5 Report/API
- UI and agents orchestrate around core, not inside core.

## Module Boundary Rules

- Keep module inputs/outputs explicit, structured, and versionable.
- Prefer pure functions and testable services.
- Avoid hidden coupling between modules.

## Data Contract Discipline

- Backward compatible changes by default.
- Any breaking schema change requires explicit version bump and migration note.
- Output schemas must be documented before broad refactors.

## AutoResearch Rule

AutoResearch is evidence synthesis and rationale generation only.
It must not replace deterministic scientific scoring logic.

## Quality Rules

- Core code should be production-structured and testable.
- Keep logic callable without web app dependencies.
- Add/update tests for major inference paths and schema outputs


This is not a generic app. It is a long-term computational oncology platform. Current state: - NeoVax has been extracted into backend/core/neoantigen/ with parity focus. - Deterministic behavior must be preserved. - NeoVax is Module 1 of a larger pipeline: M0 QC -> M1 NeoVax -> M2 CloneEscape -> M3 ResistanceLoop -> M4 TrialMatch -> M5 Report/API

 Non-negotiables: 
- Do not add UI unless explicitly asked. 
- Do not move core logic into app or CLI layers. 
- Do not silently change schemas, formulas, or ranking behavior. 
- Prefer small, test-backed edits over broad rewrites. 
- If a task affects architecture, summarize the plan before coding.
 - Preserve backward compatibility unless explicitly version-bumped.

 Coding rules: 
- Keep scientific logic in backend/core/ 
- Keep CLI thin 
- Add or update tests with meaningful changes
 - Surface assumptions explicitly 
- When uncertain, choose the smallest safe change Current priority:
 1. docs/MIGRATION_NOTES.md
 2. minimal CLI spine 
3. first real TCGA run 
4. only then expand.
