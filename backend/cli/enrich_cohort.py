"""
Enrich ``qualified_candidates.parquet`` with TCGA RNA TPM, purity, and naive CCF.

Example::

    python -m backend.cli.enrich_cohort --stub
    python -m backend.cli.enrich_cohort --pull --stub
    python -m backend.cli.enrich_cohort --metadata data/intermediate/tcga_sarc_metadata.parquet
    python -m backend.cli.enrich_cohort --stub --purity-file path/to/purity.csv
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# Allow `python backend/cli/enrich_cohort.py` (no -m): repo root on sys.path before backend imports.
if __package__ is None:
    _repo = Path(__file__).resolve().parents[2]
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))

import pandas as pd
from tqdm import tqdm

from backend.core.data.tcga_data import (
    default_metadata_path,
    join_tcga_metadata,
    load_tcga_metadata,
    run_tcga_pull_subprocess,
    write_stub_tcga_metadata,
)
from backend.core.prioritization.resistance_loop import apply_resistance_loop
from backend.core.qualification.purity_resolution import (
    apply_evidence_provenance_columns,
    apply_purity_resolution,
    load_purity_table,
    parse_priority_arg,
    parse_purity_key_arg,
)
from backend.core.qualification.real_clonality import apply_layer as apply_real_clonality_layer
from backend.core.qualification.real_expression import apply_layer as apply_real_expression_layer
from neoresist.config import get_app_config
from neoresist.metadata import stamp_enrichment_metadata


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    root = _repo_root()
    p = argparse.ArgumentParser(description="NeoResist-MD: TCGA enrichment for ResistanceLoop candidates.")
    p.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "final" / "qualified_candidates.parquet",
        help="Input qualified candidates parquet.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "final" / "enriched_candidates.parquet",
        help="Output enriched parquet.",
    )
    p.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="TCGA metadata parquet (default: data/intermediate/tcga_sarc_metadata.parquet).",
    )
    p.add_argument(
        "--intermediate-dir",
        type=Path,
        default=root / "data" / "intermediate",
        help="Directory for TCGA pulls and PyClone stub TSVs.",
    )
    p.add_argument(
        "--pull",
        action="store_true",
        help="Run R pull script before enrichment (requires Rscript + TCGAbiolinks for non-stub).",
    )
    p.add_argument(
        "--stub",
        action="store_true",
        help="Use stub TCGA metadata (Python or R --stub); no GDC download.",
    )
    p.add_argument(
        "--rscript",
        type=str,
        default="Rscript",
        help="Rscript executable for --pull.",
    )
    p.add_argument(
        "--skip-resistance-loop",
        action="store_true",
        help="Do not recompute rl_priority / tier after enrichment.",
    )
    p.add_argument(
        "--purity-file",
        type=Path,
        default=None,
        help="Optional cohort purity table (CSV, TSV, Parquet, JSON).",
    )
    p.add_argument(
        "--purity-key",
        type=str,
        default=None,
        help="Match key order: auto (default), patient_id, or barcode.",
    )
    p.add_argument(
        "--purity-source-priority",
        type=str,
        default=None,
        help="Comma list: user_file,tcga_metadata,computed_estimate,stub_fallback (default order).",
    )
    p.add_argument(
        "--purity-stub-fallback",
        action="store_true",
        help="When no purity can be resolved, use the built-in stub value (0.82). "
        "Default: leave unresolved (NaN) to preserve Phase 4 behavior when TCGA purity is absent.",
    )
    p.add_argument(
        "--scoring-profile",
        type=str,
        default=None,
        help="Scoring profile id (configs/scoring_profiles/<id>.yaml). Default: app_config.yaml.",
    )
    p.add_argument(
        "--rule-profile",
        type=str,
        default=None,
        help="Rule profile id (configs/rule_profiles/<id>.yaml). Default: app_config.yaml.",
    )
    p.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset label stored in output metadata (default: from app_config defaults.dataset_id).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _repo_root()
    inp: Path = args.input
    if not inp.is_file():
        print(f"ERROR: Input not found: {inp}", file=sys.stderr)
        return 2

    df = pd.read_parquet(inp, engine="pyarrow")
    if df.empty:
        print("ERROR: Input parquet is empty.", file=sys.stderr)
        return 3

    inter: Path = args.intermediate_dir
    inter.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.metadata) if args.metadata is not None else default_metadata_path()

    pids = sorted(df["patient_id"].dropna().astype(str).unique().tolist())
    gcol = "gene" if "gene" in df.columns else "gene_name"
    ug = df[gcol].dropna().astype(str).unique().tolist() if gcol in df.columns else []
    genes = sorted(set(ug))[:8000]

    if args.pull:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as tmp:
            tmp.write("\n".join(pids))
            plist = Path(tmp.name)
        try:
            code = run_tcga_pull_subprocess(
                out_dir=inter,
                patient_list_file=plist,
                stub=bool(args.stub),
                rscript_executable=args.rscript,
            )
        finally:
            plist.unlink(missing_ok=True)
        if code != 0 and not meta_path.is_file():
            print("Pull failed; writing Python stub metadata as fallback.", file=sys.stderr)
            write_stub_tcga_metadata(pids, meta_path, genes=genes or None)
        elif code != 0:
            print(f"WARNING: R pull exited {code}; using existing metadata if present.", file=sys.stderr)
    elif args.stub or not meta_path.is_file():
        print("Writing stub TCGA metadata (Python)…")
        write_stub_tcga_metadata(pids, meta_path, genes=genes or None)

    if not meta_path.is_file():
        print(f"ERROR: TCGA metadata not found at {meta_path}", file=sys.stderr)
        return 4

    user_purity_df: pd.DataFrame | None = None
    if args.purity_file is not None:
        ppath = Path(args.purity_file)
        if not ppath.is_file():
            print(f"ERROR: --purity-file not found: {ppath}", file=sys.stderr)
            return 5
        user_purity_df = load_purity_table(ppath)

    try:
        priority = parse_priority_arg(args.purity_source_priority)
        purity_key = parse_purity_key_arg(args.purity_key)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 6

    n_steps = 7 if args.skip_resistance_loop else 8
    pbar = tqdm(total=n_steps, desc="Enrich cohort", unit="step")
    meta = load_tcga_metadata(meta_path)
    pbar.update(1)
    merged = join_tcga_metadata(df, meta)
    pbar.update(1)
    merged = apply_purity_resolution(
        merged,
        user_purity_df=user_purity_df,
        priority=priority,
        stub_fallback=bool(args.purity_stub_fallback),
        purity_key=purity_key,
    )
    pbar.update(1)
    merged = apply_real_expression_layer(merged)
    pbar.update(1)
    pyclone_dir = inter / "pyclone_stub_inputs"
    merged = apply_real_clonality_layer(merged, pyclone_out_dir=pyclone_dir)
    pbar.update(1)

    if not args.skip_resistance_loop:
        merged = apply_resistance_loop(
            merged,
            prefer_real_evidence=True,
            scoring_profile_id=args.scoring_profile,
            rule_profile_id=args.rule_profile,
        )
        pbar.update(1)

    merged = apply_evidence_provenance_columns(merged)
    pbar.update(1)

    cfg = get_app_config()
    ds_name = args.dataset_name or cfg.defaults.dataset_id
    merged = stamp_enrichment_metadata(merged, dataset_name=ds_name)

    out: Path = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    pbar.update(1)
    pbar.close()
    print(f"Wrote {len(merged)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
