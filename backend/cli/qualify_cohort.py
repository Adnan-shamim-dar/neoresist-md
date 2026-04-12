from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backend.core.generation.aggregate_hla import aggregate_runs, discover_run_dirs, write_multi_hla_parquet
from backend.core.prioritization.resistance_loop import apply_resistance_loop
from backend.core.qualification.clonality import apply_layer as apply_clonality_layer
from backend.core.qualification.escape import apply_layer as apply_escape_layer
from backend.core.qualification.expression import apply_layer as apply_expression_layer
from backend.core.qualification.presentation import apply_layer as apply_presentation_layer


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    root = _default_repo_root()
    parser = argparse.ArgumentParser(
        description="ResistanceLoop: aggregate HLA runs, stub-qualify, and prioritize candidates.",
        epilog=(
            "Run from the neovax repo root so `import backend` resolves to this package. "
            "Point --hla-runs-root at `data/neovaxruns/hla_test` after creating a directory "
            "junction/symlink from `data/neovaxruns` to `neoresist-md/data/neovaxruns` "
            "(Windows example: mklink /J data\\neovaxruns neoresist-md\\data\\neovaxruns)."
        ),
    )
    parser.add_argument(
        "--hla-runs-root",
        type=Path,
        default=root / "data" / "neovaxruns" / "hla_test",
        help="Root containing <HLA>/<run>/candidates.csv (default: <repo>/data/neovaxruns/hla_test).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Top N candidates per (patient_id, hla_allele) by priority_score (default: 50).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "data" / "final",
        help="Directory for Parquet outputs (default: <repo>/data/final).",
    )
    parser.add_argument(
        "--only-aggregate",
        action="store_true",
        help="Only write multi_hla_candidates_v1.parquet and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    hla_root: Path = args.hla_runs_root
    if not hla_root.is_dir():
        print(
            f"ERROR: HLA runs root is not a directory: {hla_root}\n"
            "Create `data/neovaxruns` as a symlink/junction to `neoresist-md/data/neovaxruns`, "
            "or pass --hla-runs-root explicitly.",
            file=sys.stderr,
        )
        return 2

    if not discover_run_dirs(hla_root):
        print(
            f"ERROR: No runs with candidates.csv found under: {hla_root}",
            file=sys.stderr,
        )
        return 3

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    multi_path = out_dir / "multi_hla_candidates_v1.parquet"
    qualified_path = out_dir / "qualified_candidates.parquet"

    agg_df = aggregate_runs(hla_root, top_n=int(args.top_n))
    if agg_df.empty:
        print("ERROR: Aggregation produced an empty table.", file=sys.stderr)
        return 4

    write_multi_hla_parquet(agg_df, multi_path)
    print(f"Wrote {len(agg_df)} rows to {multi_path}")

    if args.only_aggregate:
        return 0

    df = agg_df
    df = apply_expression_layer(df)
    df = apply_presentation_layer(df)
    df = apply_clonality_layer(df)
    df = apply_escape_layer(df)
    df = apply_resistance_loop(df)

    df.to_parquet(qualified_path, index=False)
    print(f"Wrote {len(df)} rows to {qualified_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
