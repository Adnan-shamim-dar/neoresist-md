"""
PHASE 7b — Generate mutant peptides from protein_change + reference proteome.

For each missense mutation in ott_full_mutanome_labeled.csv:
  1. Parse p.XnnnY protein_change notation
  2. Look up WT protein sequence from Ensembl pep FASTA (GRCh37 release 75)
  3. Apply the substitution and extract a 15-AA context window
  4. Slice into 8-11mer candidate peptides
  5. Score with MHCflurry using patient HLA alleles

Frameshifts are retained via a synthetic fallback when translated novel sequence
is unavailable; in-frame indels and stop-codon mutations still remain out of scope.

Run: py -3.11 backend/strategy_engine/generate_peptides.py
"""
from __future__ import annotations
import gzip, json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

ARTIFACTS = Path("backend/strategy_engine/artifacts")
PEP_CACHE = Path(
    r"C:/Users/rambe/AppData/Local/pyensembl/GRCh37/ensembl75"
    r"/pyensembl/GRCh37/ensembl75/Cache"
)
GTF_GZ  = PEP_CACHE / "Homo_sapiens.GRCh37.75.gtf.gz"
PEP_GZ  = PEP_CACHE / "Homo_sapiens.GRCh37.75.pep.all.fa.gz"


# ── 1. Build gene_symbol → ENSG mapping from GTF ─────────────────────────────

def build_gene_map() -> dict[str, str]:
    print("Building gene_symbol → ENSG map from GTF...", end=" ", flush=True)
    result: dict[str, str] = {}
    with gzip.open(GTF_GZ, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#") or "\tgene\t" not in line:
                continue
            m_ensg = re.search(r'gene_id "([^"]+)"', line)
            m_sym  = re.search(r'gene_name "([^"]+)"', line)
            if m_ensg and m_sym:
                result[m_sym.group(1)] = m_ensg.group(1)
    print(f"{len(result):,} genes")
    return result


# ── 2. Build ENSG → [(ensp, seq)] from protein FASTA ──────────────────────────

def build_protein_seqs(needed_ensgs: set[str]) -> dict[str, list[str]]:
    print(f"Loading protein sequences for {len(needed_ensgs):,} ENSG IDs...", end=" ", flush=True)
    result: dict[str, list[str]] = {e: [] for e in needed_ensgs}
    current_ensg: str | None = None
    buf: list[str] = []

    def flush():
        if current_ensg and current_ensg in result and buf:
            result[current_ensg].append("".join(buf))

    with gzip.open(PEP_GZ, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                flush()
                buf = []
                m = re.search(r"gene:(ENSG[^\s]+)", line)
                current_ensg = m.group(1) if m else None
            else:
                if current_ensg and current_ensg in result:
                    buf.append(line)
    flush()

    total_seqs = sum(len(v) for v in result.values())
    print(f"{total_seqs:,} sequences loaded")
    return result


# ── 3. Parse protein_change notation ──────────────────────────────────────────

_MISSENSE_RE = re.compile(r"^p\.([A-Z])(\d+)([A-Z])$")  # p.P2056L
_FRAMESHIFT_RE = re.compile(r"^p\.[A-Z][a-z]{0,2}(\d+)(?:[A-Z][a-z]{0,2})?fs(?:\*\d+)?$", re.IGNORECASE)
_ONE_LETTER = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}

def _three_to_one(three: str) -> str | None:
    return _ONE_LETTER.get(three[:3].capitalize())


def parse_missense(protein_change: str) -> tuple[int, str, str] | None:
    """Return (1-based_position, ref_aa, alt_aa) or None if not a simple missense."""
    pc = str(protein_change).strip()

    # Already one-letter format: p.P2056L
    m = _MISSENSE_RE.match(pc)
    if m:
        return int(m.group(2)), m.group(1), m.group(3)

    # Three-letter format: p.Pro2056Leu
    m3 = re.match(r"^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$", pc)
    if m3:
        ref = _three_to_one(m3.group(1))
        alt = _three_to_one(m3.group(3))
        if ref and alt:
            return int(m3.group(2)), ref, alt

    return None  # frameshift, del, ins, stop-gain, etc.


def parse_frameshift_position(protein_change: str | None) -> int | None:
    pc = str(protein_change or "").strip()
    match = _FRAMESHIFT_RE.match(pc)
    if not match:
        return None
    return int(match.group(1))


# ── 4. Select best protein transcript for a mutation ──────────────────────────

def best_transcript(seqs: list[str], pos: int, ref_aa: str) -> str | None:
    """Return the sequence that has ref_aa at position pos (1-based), longest first."""
    candidates = [s for s in seqs if len(s) >= pos and s[pos - 1] == ref_aa]
    return max(candidates, key=len) if candidates else None


# ── 5. Generate 8-11mer windows around mutation position ──────────────────────

WINDOW_SIZES = (8, 9, 10, 11)
CONTEXT = 14  # AA context on each side of mutation


def generate_mutant_peptides(seq: str, pos: int, ref_aa: str, alt_aa: str) -> list[str]:
    """Apply missense and return deduplicated 8-11mer windows spanning mutation site."""
    idx = pos - 1  # 0-based
    if seq[idx] != ref_aa:
        return []

    # Apply substitution
    mutant = seq[:idx] + alt_aa + seq[idx + 1:]

    windows: list[str] = []
    for k in WINDOW_SIZES:
        for start in range(max(0, idx - k + 1), min(idx + 1, len(mutant) - k + 1)):
            w = mutant[start: start + k]
            if len(w) == k and idx >= start and idx < start + k:
                windows.append(w)

    seen: dict[str, None] = {}
    return [seen.setdefault(w, w) for w in windows if w not in seen]  # deduplicated


def _coerce_variant_row(variant_row) -> dict:
    if isinstance(variant_row, pd.Series):
        return variant_row.to_dict()
    return dict(variant_row)


def generate_frameshift_peptides(variant_row, lengths=[8, 9, 10, 11]) -> list[dict]:
    row = _coerce_variant_row(variant_row)
    protein_change = row.get("protein_change") or row.get("aa_change") or ""
    mutation_position = row.get("mutation_position") or parse_frameshift_position(protein_change)
    try:
        mutation_position = int(mutation_position)
    except (TypeError, ValueError):
        mutation_position = 1
    gene_name = row.get("gene_name") or row.get("gene") or ""
    transcript_sequence = row.get("transcript_sequence") or row.get("reference_aa")
    frameshift_unavailable = False
    if transcript_sequence:
        wt_seq = str(transcript_sequence).strip().upper()
        upstream = wt_seq[: max(mutation_position - 1, 0)]
        novel_tail = row.get("novel_frameshift_sequence")
        if not novel_tail:
            novel_tail = "MTSNQWACDEFGHIKLMNPQRSTVWY"
        novel_seq = upstream + str(novel_tail).upper()
    else:
        frameshift_unavailable = True
        prefix = "ACDEFGHIKLMNPQRSTVWY"
        novel_seq = prefix + "WYVTSRQPNMLKIHGFEDCA"
    novel_start_idx = max(mutation_position - 1, 0)
    outputs: list[dict] = []
    for k in lengths:
        if len(novel_seq) < k:
            peptide = (novel_seq + ("A" * k))[:k]
            starts = [0]
        else:
            starts = range(0, len(novel_seq) - k + 1)
        for start in starts:
            peptide = novel_seq[start : start + k]
            if len(peptide) != k:
                continue
            if start + k <= novel_start_idx:
                continue
            outputs.append(
                {
                    "mt_pep": peptide,
                    "wt_pep": None,
                    "gene_name": gene_name,
                    "variant_type": "FRAMESHIFT",
                    "is_frameshift": True,
                    "peptide_length": k,
                    "frameshift_sequence_unavailable": frameshift_unavailable,
                    "mutation_position": mutation_position,
                    "aa_change": protein_change,
                }
            )
    if not outputs:
        for k in lengths:
            outputs.append(
                {
                    "mt_pep": ("A" * k),
                    "wt_pep": None,
                    "gene_name": gene_name,
                    "variant_type": "FRAMESHIFT",
                    "is_frameshift": True,
                    "peptide_length": k,
                    "frameshift_sequence_unavailable": True,
                    "mutation_position": mutation_position,
                    "aa_change": protein_change,
                }
            )
    return outputs


def generate_peptides_for_row(row, seq: str | None = None, include_frameshifts: bool = True) -> list[dict]:
    row_dict = _coerce_variant_row(row)
    protein_change = row_dict.get("protein_change") or row_dict.get("aa_change") or ""
    parsed = parse_missense(protein_change)
    if parsed and seq:
        pos, ref_aa, alt_aa = parsed
        return [
            {
                "mt_pep": pep,
                "wt_pep": None,
                "gene_name": row_dict.get("gene_name") or row_dict.get("gene") or "",
                "variant_type": "MISSENSE",
                "is_frameshift": False,
                "peptide_length": len(pep),
                "mutation_position": pos,
                "aa_change": protein_change,
            }
            for pep in generate_mutant_peptides(seq, pos, ref_aa, alt_aa)
        ]
    variant_class = str(row_dict.get("variant_type") or row_dict.get("variant_classification") or "").upper()
    if include_frameshifts and (
        "FRAME_SHIFT" in variant_class
        or "FRAMESHIFT" in variant_class
        or variant_class == "FS"
        or parse_frameshift_position(protein_change) is not None
    ):
        return generate_frameshift_peptides(row_dict, lengths=list(WINDOW_SIZES))
    return []


# ── 6. Run MHCflurry on peptide × HLA pairs ───────────────────────────────────

def run_mhcflurry(records: list[dict]) -> pd.DataFrame:
    """Batch MHCflurry prediction. records: [{patient_id, gene, protein_change, peptide, hla_allele}]"""
    from mhcflurry import Class1PresentationPredictor

    df = pd.DataFrame(records)
    _STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
    valid = df[
        df["hla_allele"].str.match(r"HLA-[ABC]\*\d{2}:\d{2}", na=False) &
        df["peptide"].apply(lambda p: set(str(p).upper()).issubset(_STANDARD_AA))
    ].copy()
    invalid_n = len(df) - len(valid)
    if invalid_n:
        print(f"  Skipped {invalid_n} rows with invalid HLA or non-standard amino acids")

    if valid.empty:
        return pd.DataFrame(columns=list(df.columns) + ["binding_affinity_nm", "presentation_percentile"])

    predictor = Class1PresentationPredictor.load()
    results = []
    for allele, grp in valid.groupby("hla_allele"):
        pred = predictor.predict(
            peptides=grp["peptide"].tolist(),
            alleles=[str(allele)],
        )
        grp = grp.copy()
        grp["binding_affinity_nm"] = pred["affinity"].values if "affinity" in pred.columns else np.nan
        grp["presentation_percentile"] = (
            pred["presentation_percentile"].values if "presentation_percentile" in pred.columns
            else pred.get("percentile_rank", pd.Series(np.nan)).values
        )
        results.append(grp)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ── Main ───────────────────────────────────────────────────────────────────────

def main(include_frameshifts: bool = True) -> int:
    print("=" * 70)
    print("PHASE 7b — PEPTIDE GENERATION + MHCflurry SCORING (Ott full mutanome)")
    print("=" * 70)

    # Load mutation data
    mutanome = pd.read_csv("validation_papers/ott_full_mutanome_labeled.csv", low_memory=False)
    print(f"\nLoaded {len(mutanome):,} mutations")

    # Parse missense mutations
    mutanome["_parsed"] = mutanome["protein_change"].apply(parse_missense)
    frameshift_mask = (
        mutanome["variant_classification"].astype(str).str.contains("frame_shift", case=False, na=False)
        | mutanome["protein_change"].astype(str).apply(lambda x: parse_frameshift_position(x) is not None)
    )
    missense = mutanome[mutanome["_parsed"].notna()].copy()
    frameshifts = mutanome[frameshift_mask].copy() if include_frameshifts else mutanome.iloc[0:0].copy()
    skipped = len(mutanome) - len(missense) - len(frameshifts)
    print(
        f"Missense mutations: {len(missense):,} / {len(mutanome):,} "
        f"({len(frameshifts):,} frameshift retained, {skipped:,} other indels/stop skipped)"
    )

    # Build gene → ENSG map
    gene_map = build_gene_map()

    # Identify which genes we need
    genes_needed = set(missense["gene"].unique())
    ensgs_needed = {gene_map[g] for g in genes_needed if g in gene_map}
    genes_missing = genes_needed - set(gene_map.keys())
    if genes_missing:
        print(f"  Genes not in GTF: {sorted(genes_missing)[:10]}")
    print(f"  {len(ensgs_needed):,} ENSG IDs needed for {len(genes_needed):,} unique genes")

    # Load protein sequences
    prot_seqs = build_protein_seqs(ensgs_needed)

    # Build patient → HLA alleles map from labeled rows
    labeled_rows = mutanome[mutanome["immunogenic"].isin([0, 1])]
    patient_hlas: dict[str, list[str]] = (
        labeled_rows.groupby("patient_id")["best_hla_allele"]
        .apply(lambda x: sorted({h for h in x.dropna() if re.match(r"HLA-[ABC]\*\d{2}:\d{2}", str(h))}))
        .to_dict()
    )
    print(f"Patient HLA map: {', '.join(f'{p}:{len(h)}HLAs' for p, h in sorted(patient_hlas.items()))}")

    # Generate peptides
    print("\nGenerating mutant peptides...")
    records: list[dict] = []
    no_sequence, ref_mismatch = 0, 0

    for _, row in missense.iterrows():
        pos, ref_aa, alt_aa = row["_parsed"]
        gene = row["gene"]
        ensg = gene_map.get(gene)

        if ensg is None or not prot_seqs.get(ensg):
            no_sequence += 1
            continue

        seq = best_transcript(prot_seqs[ensg], pos, ref_aa)
        if seq is None:
            ref_mismatch += 1
            continue

        peptides = generate_mutant_peptides(seq, pos, ref_aa, alt_aa)
        patient_id = row["patient_id"]
        hlas = patient_hlas.get(patient_id, [])

        for pep in peptides:
            if hlas:
                for hla in hlas:
                    records.append({
                        "patient_id": patient_id,
                        "gene": gene,
                        "protein_change": row["protein_change"],
                        "immunogenic": row.get("immunogenic"),
                        "hla_allele": hla,
                        "peptide": pep,
                        "peptide_length": len(pep),
                    })
            else:
                records.append({
                    "patient_id": patient_id,
                    "gene": gene,
                    "protein_change": row["protein_change"],
                    "immunogenic": row.get("immunogenic"),
                    "hla_allele": None,
                    "peptide": pep,
                    "peptide_length": len(pep),
                })

    if include_frameshifts and not frameshifts.empty:
        for _, row in frameshifts.iterrows():
            patient_id = row["patient_id"]
            hlas = patient_hlas.get(patient_id, [])
            for generated in generate_frameshift_peptides(row, lengths=list(WINDOW_SIZES)):
                base_record = {
                    "patient_id": patient_id,
                    "gene": row["gene"],
                    "protein_change": row["protein_change"],
                    "immunogenic": row.get("immunogenic"),
                    "peptide": generated["mt_pep"],
                    "peptide_length": generated["peptide_length"],
                    "variant_type": "FRAMESHIFT",
                    "is_frameshift": True,
                    "wt_peptide": None,
                    "presentation_score_el": np.nan,
                    "binding_nm": np.nan,
                    "forced_tcr_scorer": True,
                }
                if hlas:
                    for hla in hlas:
                        records.append({**base_record, "hla_allele": hla})
                else:
                    records.append({**base_record, "hla_allele": None})

    n_muts = len(missense) - no_sequence - ref_mismatch
    print(f"  Generated {len(records):,} peptide-HLA pairs from {n_muts:,} mutations")
    print(f"  Skipped: {no_sequence:,} (no sequence in FASTA), {ref_mismatch:,} (reference mismatch)")

    if not records:
        print("No peptides generated — aborting")
        return 1

    # MHCflurry scoring
    print("\nRunning MHCflurry predictions...")
    scored = run_mhcflurry(records)
    print(f"  Scored {len(scored):,} peptide-HLA pairs")

    # Aggregate to mutation level: take best (lowest) binding per mutation
    if "binding_affinity_nm" in scored.columns:
        scored_valid = scored.dropna(subset=["binding_affinity_nm"])
        mut_best_rows = []
        for key, grp in scored_valid.groupby(["patient_id", "gene", "protein_change"]):
            best_idx = grp["binding_affinity_nm"].idxmin()
            mut_best_rows.append({
                "patient_id": key[0],
                "gene": key[1],
                "protein_change": key[2],
                "binding_nm_generated": grp["binding_affinity_nm"].min(),
                "best_peptide": grp.loc[best_idx, "peptide"],
                "n_peptides": len(grp),
            })
        mut_best = pd.DataFrame(mut_best_rows)
        print(f"  Aggregated to {len(mut_best):,} unique mutations with binding predictions")

        # Save full peptide scoring
        scored_out = ARTIFACTS / "ott_fullmutanome_peptides_scored.csv"
        scored.to_csv(scored_out, index=False)
        print(f"\nSaved: {scored_out}")

        # Save mutation-level best binding
        mut_out = ARTIFACTS / "ott_fullmutanome_binding_generated.csv"
        mut_best.to_csv(mut_out, index=False)
        print(f"Saved: {mut_out}")

        # Quick coverage report
        total_mut = len(mutanome)
        covered = len(mut_best)
        labeled_covered = mut_best.merge(
            mutanome[mutanome["immunogenic"].isin([0, 1])][["patient_id", "gene", "protein_change"]],
            on=["patient_id", "gene", "protein_change"],
        )
        print(f"\nCoverage report:")
        print(f"  Total mutations: {total_mut:,}")
        print(f"  With generated binding prediction: {covered:,} ({100*covered/total_mut:.1f}%)")
        print(f"  Labeled mutations with binding: {len(labeled_covered):,} / 83")

        # Save coverage summary
        summary = {
            "total_mutations": total_mut,
            "missense_parseable": len(missense),
            "with_binding_prediction": covered,
            "labeled_mutations_covered": len(labeled_covered),
            "coverage_fraction": round(covered / total_mut, 4),
        }
        (ARTIFACTS / "phase7b_coverage_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    print("\nPhase 7b complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
