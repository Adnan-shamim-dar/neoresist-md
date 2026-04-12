from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class Provenance(BaseModel):
    patient_id: str
    hla_allele: str
    run_timestamp: str | None = None


class Mutation(BaseModel):
    gene: str
    protein_change: str | None = None
    vaf: float | None = None
    ref_count: int | None = None
    alt_count: int | None = None


class Peptide(BaseModel):
    mutant_peptide: str
    wt_peptide: str | None = None
    length: int = Field(ge=0)


class Binding(BaseModel):
    affinity_nm: float | None = None
    percentile_rank: float | None = None


class Evidence(BaseModel):
    expression_tpm: float | None = None
    presentation_score: float | None = Field(default=None, ge=0, le=1)
    ccf: float | None = Field(default=None, ge=0, le=1)
    hla_loh_flag: str | None = None
    self_dissimilarity: float | None = Field(default=None, ge=0, le=1)


class Final(BaseModel):
    tier: int = Field(ge=1, le=3)
    rl_priority: float | None = None
    exclusion_reasons: list[str] = Field(default_factory=list)

    @field_validator("exclusion_reasons", mode="before")
    @classmethod
    def _coerce_exclusions(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v]
        return [str(v)]


class CandidateRecord(BaseModel):
    provenance: Provenance
    mutation: Mutation
    peptide: Peptide
    binding: Binding
    evidence: Evidence
    final: Final


def flatten_candidate(record: CandidateRecord) -> dict[str, Any]:
    """Flatten nested model to a single dict suitable for pandas/Parquet."""
    p, m, pep, b, e, f = (
        record.provenance,
        record.mutation,
        record.peptide,
        record.binding,
        record.evidence,
        record.final,
    )
    return {
        "patient_id": p.patient_id,
        "hla_allele": p.hla_allele,
        "run_timestamp": p.run_timestamp,
        "gene": m.gene,
        "protein_change": m.protein_change,
        "vaf": m.vaf,
        "ref_count": m.ref_count,
        "alt_count": m.alt_count,
        "mutant_peptide": pep.mutant_peptide,
        "wt_peptide": pep.wt_peptide,
        "peptide_length": pep.length,
        "affinity_nm": b.affinity_nm,
        "percentile_rank": b.percentile_rank,
        "expression_tpm": e.expression_tpm,
        "presentation_score": e.presentation_score,
        "ccf": e.ccf,
        "hla_loh_flag": e.hla_loh_flag,
        "self_dissimilarity": e.self_dissimilarity,
        "tier": f.tier,
        "rl_priority": f.rl_priority,
        "exclusion_reasons": list(f.exclusion_reasons),
    }


def validate_candidate_row(row: dict[str, Any]) -> CandidateRecord:
    """Build and validate a CandidateRecord from a flat or nested row dict."""
    if "provenance" in row:
        return CandidateRecord.model_validate(row)
    return CandidateRecord(
        provenance=Provenance(
            patient_id=row["patient_id"],
            hla_allele=row["hla_allele"],
            run_timestamp=row.get("run_timestamp"),
        ),
        mutation=Mutation(
            gene=row["gene"],
            protein_change=row.get("protein_change"),
            vaf=row.get("vaf"),
            ref_count=row.get("ref_count"),
            alt_count=row.get("alt_count"),
        ),
        peptide=Peptide(
            mutant_peptide=row["mutant_peptide"],
            wt_peptide=row.get("wt_peptide"),
            length=int(row.get("peptide_length", len(row.get("mutant_peptide", "")))),
        ),
        binding=Binding(
            affinity_nm=row.get("affinity_nm"),
            percentile_rank=row.get("percentile_rank"),
        ),
        evidence=Evidence(
            expression_tpm=row.get("expression_tpm"),
            presentation_score=row.get("presentation_score"),
            ccf=row.get("ccf"),
            hla_loh_flag=row.get("hla_loh_flag"),
            self_dissimilarity=row.get("self_dissimilarity"),
        ),
        final=Final(
            tier=int(row.get("tier", 3)),
            rl_priority=row.get("rl_priority"),
            exclusion_reasons=row.get("exclusion_reasons") or [],
        ),
    )
