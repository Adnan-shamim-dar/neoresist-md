from __future__ import annotations

import pandas as pd
from mhcflurry import Class1PresentationPredictor


REQUIRED_COLUMNS = {
    "gene_name",
    "protein_sequence",
    "mutation_position",
    "normal_amino_acid",
    "mutant_amino_acid",
}
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
PEPTIDE_LENGTHS = (8, 9, 10, 11)


def validate_mutation_rows(df: pd.DataFrame) -> None:
    for i, row in enumerate(df.itertuples(index=False), start=1):
        sequence = str(row.protein_sequence).strip().upper()
        normal_aa = str(row.normal_amino_acid).strip().upper()
        mutant_aa = str(row.mutant_amino_acid).strip().upper()

        if not sequence:
            raise ValueError(
                f"Row {i}: protein_sequence is empty. Provide a valid amino acid sequence."
            )

        try:
            mutation_position = int(row.mutation_position)
        except (TypeError, ValueError):
            raise ValueError(
                f"Row {i}: mutation_position must be an integer (1-based indexing)."
            ) from None

        if mutation_position < 1 or mutation_position > len(sequence):
            raise ValueError(
                f"Row {i}: mutation_position={mutation_position} is out of range for "
                f"protein length {len(sequence)} (valid: 1..{len(sequence)})."
            )

        if len(normal_aa) != 1 or normal_aa not in VALID_AMINO_ACIDS:
            raise ValueError(
                f"Row {i}: normal_amino_acid must be one valid amino acid "
                f"(allowed: {''.join(sorted(VALID_AMINO_ACIDS))})."
            )

        if len(mutant_aa) != 1 or mutant_aa not in VALID_AMINO_ACIDS:
            raise ValueError(
                f"Row {i}: mutant_amino_acid must be one valid amino acid "
                f"(allowed: {''.join(sorted(VALID_AMINO_ACIDS))})."
            )

        sequence_aa = sequence[mutation_position - 1]
        if sequence_aa != normal_aa:
            raise ValueError(
                f"Row {i}: normal_amino_acid='{normal_aa}' does not match "
                f"protein_sequence at 1-based position {mutation_position} "
                f"(found '{sequence_aa}')."
            )

        if mutant_aa == normal_aa:
            raise ValueError(
                f"Row {i}: mutant_amino_acid must be different from normal_amino_acid."
            )


def _validate_input(df: pd.DataFrame, hla_alleles: list[str]) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required columns: {', '.join(sorted(missing))}"
        )
    if not hla_alleles:
        raise ValueError("hla_alleles must contain at least one HLA allele.")


def _mutant_sequence(sequence: str, position_1_based: int, mutant_aa: str) -> str:
    idx = position_1_based - 1
    if idx < 0 or idx >= len(sequence):
        raise ValueError(f"Mutation position {position_1_based} is out of range.")
    return sequence[:idx] + mutant_aa + sequence[idx + 1 :]


def _spanning_mutant_peptides(mutant_sequence: str, position_1_based: int) -> list[str]:
    idx = position_1_based - 1
    peptides: set[str] = set()
    for k in PEPTIDE_LENGTHS:
        start_min = max(0, idx - k + 1)
        start_max = min(idx, len(mutant_sequence) - k)
        if start_min > start_max:
            continue
        for start in range(start_min, start_max + 1):
            peptide = mutant_sequence[start : start + k]
            if len(peptide) == k:
                peptides.add(peptide)
    return sorted(peptides)


def _rank_to_score(series: pd.Series, ascending: bool) -> pd.Series:
    ranks = series.rank(method="min", ascending=ascending)
    n = len(ranks)
    if n <= 1:
        return pd.Series([1.0] * n, index=series.index)
    return (n - ranks) / (n - 1)


def _ranking_reason(presentation_component: float, affinity_component: float) -> str:
    strong_threshold = 0.67
    weak_threshold = 0.33
    presentation_label = (
        "strong presentation"
        if presentation_component >= strong_threshold
        else "weak presentation"
        if presentation_component <= weak_threshold
        else "moderate presentation"
    )
    affinity_label = (
        "strong affinity"
        if affinity_component >= strong_threshold
        else "weak affinity"
        if affinity_component <= weak_threshold
        else "moderate affinity"
    )
    return f"{presentation_label} + {affinity_label}"


def _triage_label(presentation_component: float, affinity_component: float) -> str:
    if presentation_component >= 0.75 and affinity_component >= 0.60:
        return "High Priority"
    if presentation_component >= 0.50 and affinity_component >= 0.40:
        return "Medium Priority"
    return "Low Priority"


def _triage_note(triage_label: str, ranking_reason: str) -> str:
    if triage_label == "High Priority":
        return f"Heuristic high-priority candidate: {ranking_reason}."
    if triage_label == "Medium Priority":
        return f"Heuristic medium-priority candidate: {ranking_reason}."
    return f"Heuristic low-priority candidate: {ranking_reason}."


def predict_candidates(df: pd.DataFrame, hla_alleles: list[str]) -> pd.DataFrame:
    _validate_input(df, hla_alleles)
    validate_mutation_rows(df)

    peptide_rows: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        gene_name = str(row.gene_name)
        sequence = str(row.protein_sequence).strip().upper()
        mutation_position = int(row.mutation_position)
        mutant_aa = str(row.mutant_amino_acid).strip().upper()

        mutant_sequence = _mutant_sequence(sequence, mutation_position, mutant_aa)
        for peptide in _spanning_mutant_peptides(mutant_sequence, mutation_position):
            peptide_rows.append(
                {
                    "gene_name": gene_name,
                    "mutation_position": mutation_position,
                    "peptide": peptide,
                }
            )

    if not peptide_rows:
        return pd.DataFrame(
            columns=[
                "gene_name",
                "mutation_position",
                "peptide",
                "best_allele",
                "presentation_score",
                "presentation_percentile",
                "affinity",
                "presentation_component",
                "affinity_component",
                "priority_score",
                "ranking_reason",
                "triage_label",
                "triage_note",
            ]
        )

    peptide_df = pd.DataFrame(peptide_rows).drop_duplicates()
    unique_peptides = sorted(peptide_df["peptide"].unique().tolist())

    predictor = Class1PresentationPredictor.load()
    raw_predictions = predictor.predict(peptides=unique_peptides, alleles=hla_alleles)
    best_by_peptide = raw_predictions.copy()

    merged = peptide_df.merge(
        best_by_peptide[
            ["peptide", "best_allele", "presentation_score", "presentation_percentile", "affinity"]
        ],
        on="peptide",
        how="left",
    )

    merged["presentation_component"] = _rank_to_score(merged["presentation_score"], ascending=False)
    merged["affinity_component"] = _rank_to_score(merged["affinity"], ascending=True)
    merged["priority_score"] = 0.6 * merged["presentation_component"] + 0.4 * merged["affinity_component"]
    merged["ranking_reason"] = merged.apply(
        lambda row: _ranking_reason(float(row["presentation_component"]), float(row["affinity_component"])),
        axis=1,
    )
    merged["triage_label"] = merged.apply(
        lambda row: _triage_label(float(row["presentation_component"]), float(row["affinity_component"])),
        axis=1,
    )
    merged["triage_note"] = merged.apply(
        lambda row: _triage_note(str(row["triage_label"]), str(row["ranking_reason"])),
        axis=1,
    )

    final = merged[
        [
            "gene_name",
            "mutation_position",
            "peptide",
            "best_allele",
            "presentation_score",
            "presentation_percentile",
            "affinity",
            "presentation_component",
            "affinity_component",
            "priority_score",
            "ranking_reason",
            "triage_label",
            "triage_note",
        ]
    ].sort_values(by=["priority_score", "presentation_score", "affinity"], ascending=[False, False, True])
    return final.reset_index(drop=True)
