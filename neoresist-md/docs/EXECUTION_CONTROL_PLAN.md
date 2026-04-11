# EXECUTION_CONTROL_PLAN

## Purpose

Persistent execution tracker for implementing the NeoResist-MD plan without losing context across chats.

## Operating Mode

- Perplexity plan = strategic authority.
- This execution layer = implementation authority.
- No core scientific code changes without explicit user confirmation.
- Documentation/scaffolding/schema planning can proceed immediately.

## Current State Snapshot

- Repo skeleton exists for `neoresist-md/`.
- Canonical plan exists in `docs/NEORESIST-MD_MASTER_PLAN.md`.
- Architecture and I/O contracts exist in `docs/ARCHITECTURE.md`.
- Placeholder module folders exist under `backend/core/`.

## Phase Plan

### Phase A - Documentation Lock (in progress)

- [x] Canonical master plan file
- [x] Architecture with module contracts
- [x] Roadmap file
- [x] Cursor system prompt file
- [x] Execution control plan file

### Phase B - Schema Lock (approval needed for implementation details)

- [ ] Add JSON schema specs for M0-M4 outputs
- [ ] Add schema versioning policy examples
- [ ] Add module-level TODO files

### Phase C - Core Spine (requires explicit approval before coding)

- [ ] Extract NeoVax core service into `backend/core/neoantigen/`
- [ ] Add QC placeholder module
- [ ] Add CLI pipeline runner
- [ ] Emit schema-valid outputs across modules

### Phase D - Validation and Expansion

- [ ] Add retrospective validation cases
- [ ] Add audit/confidence reporting
- [ ] Add thin web/API integration layer

## Decision Gates (must confirm with user)

- Gate 1: start core extraction from existing codebase
- Gate 2: choose schema format/tooling
- Gate 3: choose first validation cohort and metrics
- Gate 4: expose API/FHIR-facing artifacts

## Context Continuity Rules

- Every major change updates:
  - `docs/ROADMAP.md`
  - `docs/ARCHITECTURE.md` (if contracts changed)
  - this file (`docs/EXECUTION_CONTROL_PLAN.md`)
- Keep a short change note for each architecture decision.
- Treat this file as primary implementation memory between chats.
