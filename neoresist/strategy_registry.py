from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from neoresist.paths import audit_log_path, strategy_store_dir
from neoresist.profiles import load_rule_profile, load_scoring_profile

SAFE_WEIGHT_KEYS = ("expression_norm", "presentation", "ccf", "self_dissimilarity")


class StrategyDefinitionContract(BaseModel):
    schema_version: str = "1.0.0"
    strategy_id: str
    display_name: str
    description: str = ""
    origin: str = "builtin"
    parent_strategy_id: str | None = None
    scoring_profile_id: str
    rule_profile_id: str
    strategy_version: str
    created_at: str | None = None
    last_modified: str | None = None
    expression_tpm_cap: float = Field(ge=1.0)
    blend_expression: float = Field(ge=0.0, le=1.0)
    blend_ccf: float = Field(ge=0.0, le=1.0)
    escape_penalty_weight: float = Field(ge=-1.0, le=1.0)
    tier1_above: float = Field(ge=0.0, le=1.0)
    tier2_above: float = Field(ge=0.0, le=1.0)
    weights: dict[str, float]


class AuditEventContract(BaseModel):
    schema_version: str = "1.0.0"
    timestamp: str
    event_type: str
    strategy_id: str
    strategy_origin: str
    parent_strategy_id: str | None = None
    app_version: str
    scoring_profile_id: str
    scoring_version: str
    rule_profile_id: str
    changes: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CoherenceResultContract(BaseModel):
    schema_version: str = "1.0.0"
    candidate_key: str
    strategy_id: str
    coherence_score: float = Field(ge=0.0, le=1.0)
    coherence_reasons: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class ResolvedStrategy:
    strategy_id: str
    display_name: str
    description: str
    origin: str
    parent_strategy_id: str | None
    scoring_profile_id: str
    scoring_version: str
    rule_profile_id: str
    rule_version: str
    strategy_version: str
    created_at: str | None
    last_modified: str | None
    expression_tpm_cap: float
    blend_expression: float
    blend_ccf: float
    weights: dict[str, float]
    escape_penalty_weight: float
    tier1_above: float
    tier2_above: float


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _slugify(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return clean or "strategy"


def _ensure_state_dirs() -> None:
    strategy_store_dir().mkdir(parents=True, exist_ok=True)
    audit_log_path().parent.mkdir(parents=True, exist_ok=True)


def _normalize_weights(weights: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in SAFE_WEIGHT_KEYS:
        val = weights.get(key, 0.0)
        try:
            out[key] = max(0.0, min(1.0, float(val)))
        except (TypeError, ValueError):
            out[key] = 0.0
    return out


def strategy_from_profiles(scoring_profile_id: str, rule_profile_id: str = "default_rules") -> ResolvedStrategy:
    sp = load_scoring_profile(scoring_profile_id)
    rp = load_rule_profile(rule_profile_id)
    return ResolvedStrategy(
        strategy_id=sp.profile_id,
        display_name=sp.display_name,
        description=sp.description,
        origin="builtin",
        parent_strategy_id=None,
        scoring_profile_id=sp.profile_id,
        scoring_version=sp.version,
        rule_profile_id=rp.profile_id,
        rule_version=rp.version,
        strategy_version=f"{sp.version}+{rp.version}",
        created_at=None,
        last_modified=None,
        expression_tpm_cap=float(sp.expression_tpm_cap),
        blend_expression=float(sp.blend_expression),
        blend_ccf=float(sp.blend_ccf),
        weights=_normalize_weights(sp.weights),
        escape_penalty_weight=float(sp.escape_penalty_weight),
        tier1_above=float(rp.tier1_above),
        tier2_above=float(rp.tier2_above),
    )


def _strategy_file_path(strategy_id: str) -> Path:
    return strategy_store_dir() / f"{strategy_id}.json"


def _strategy_to_contract(strategy: ResolvedStrategy) -> StrategyDefinitionContract:
    return StrategyDefinitionContract(
        strategy_id=strategy.strategy_id,
        display_name=strategy.display_name,
        description=strategy.description,
        origin=strategy.origin,
        parent_strategy_id=strategy.parent_strategy_id,
        scoring_profile_id=strategy.scoring_profile_id,
        rule_profile_id=strategy.rule_profile_id,
        strategy_version=strategy.strategy_version,
        created_at=strategy.created_at,
        last_modified=strategy.last_modified,
        expression_tpm_cap=strategy.expression_tpm_cap,
        blend_expression=strategy.blend_expression,
        blend_ccf=strategy.blend_ccf,
        escape_penalty_weight=strategy.escape_penalty_weight,
        tier1_above=strategy.tier1_above,
        tier2_above=strategy.tier2_above,
        weights=_normalize_weights(strategy.weights),
    )


def _contract_to_strategy(contract: StrategyDefinitionContract) -> ResolvedStrategy:
    sp = load_scoring_profile(contract.scoring_profile_id)
    rp = load_rule_profile(contract.rule_profile_id)
    return ResolvedStrategy(
        strategy_id=contract.strategy_id,
        display_name=contract.display_name,
        description=contract.description,
        origin=contract.origin,
        parent_strategy_id=contract.parent_strategy_id,
        scoring_profile_id=contract.scoring_profile_id,
        scoring_version=sp.version,
        rule_profile_id=contract.rule_profile_id,
        rule_version=rp.version,
        strategy_version=contract.strategy_version,
        created_at=contract.created_at,
        last_modified=contract.last_modified,
        expression_tpm_cap=float(contract.expression_tpm_cap),
        blend_expression=float(contract.blend_expression),
        blend_ccf=float(contract.blend_ccf),
        weights=_normalize_weights(contract.weights),
        escape_penalty_weight=float(contract.escape_penalty_weight),
        tier1_above=float(contract.tier1_above),
        tier2_above=float(contract.tier2_above),
    )


def list_strategies() -> list[ResolvedStrategy]:
    _ensure_state_dirs()
    config_root = Path(__file__).resolve().parent.parent / "configs"
    builtin_ids = sorted(
        {
            p.stem
            for subdir in ("scoring_profiles", "profiles")
            for p in (config_root / subdir).glob("*.yaml")
        }
    )
    items = [strategy_from_profiles(profile_id) for profile_id in builtin_ids]
    seen_ids = {item.strategy_id for item in items}
    for path in sorted(strategy_store_dir().glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            item = _contract_to_strategy(StrategyDefinitionContract.model_validate(raw))
            if item.strategy_id in seen_ids:
                continue
            items.append(item)
            seen_ids.add(item.strategy_id)
        except Exception:
            continue
    items.sort(key=lambda s: (0 if s.origin == "builtin" else 1, s.display_name.lower()))
    return items


def get_strategy(strategy_id: str) -> ResolvedStrategy:
    for item in list_strategies():
        if item.strategy_id == strategy_id:
            return item
    raise KeyError(strategy_id)


def clone_strategy(source_id: str, *, display_name: str | None = None) -> ResolvedStrategy:
    src = get_strategy(source_id)
    now = _now_iso()
    base_name = display_name or f"{src.display_name} Clone"
    return ResolvedStrategy(
        strategy_id=f"user-{_slugify(base_name)}",
        display_name=base_name,
        description=src.description,
        origin="user_variant",
        parent_strategy_id=src.strategy_id,
        scoring_profile_id=src.scoring_profile_id,
        scoring_version=src.scoring_version,
        rule_profile_id=src.rule_profile_id,
        rule_version=src.rule_version,
        strategy_version="1.0",
        created_at=now,
        last_modified=now,
        expression_tpm_cap=src.expression_tpm_cap,
        blend_expression=src.blend_expression,
        blend_ccf=src.blend_ccf,
        weights=dict(src.weights),
        escape_penalty_weight=src.escape_penalty_weight,
        tier1_above=src.tier1_above,
        tier2_above=src.tier2_above,
    )


def save_strategy(strategy: ResolvedStrategy) -> ResolvedStrategy:
    _ensure_state_dirs()
    current = strategy
    now = _now_iso()
    if not current.created_at:
        current = ResolvedStrategy(**{**asdict(current), "created_at": now})
    current = ResolvedStrategy(**{**asdict(current), "last_modified": now})
    path = _strategy_file_path(current.strategy_id)
    path.write_text(_strategy_to_contract(current).model_dump_json(indent=2), encoding="utf-8")
    return current


def serialize_strategy(strategy: ResolvedStrategy) -> dict[str, Any]:
    return asdict(strategy)


def deserialize_strategy(data: dict[str, Any]) -> ResolvedStrategy:
    return ResolvedStrategy(**data)


def log_audit_event(
    *,
    event_type: str,
    strategy: ResolvedStrategy,
    app_version: str,
    changes: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    _ensure_state_dirs()
    event = AuditEventContract(
        timestamp=_now_iso(),
        event_type=event_type,
        strategy_id=strategy.strategy_id,
        strategy_origin=strategy.origin,
        parent_strategy_id=strategy.parent_strategy_id,
        app_version=app_version,
        scoring_profile_id=strategy.scoring_profile_id,
        scoring_version=strategy.scoring_version,
        rule_profile_id=strategy.rule_profile_id,
        changes=changes or {},
        metadata=metadata or {},
    )
    with audit_log_path().open("a", encoding="utf-8") as fh:
        fh.write(event.model_dump_json())
        fh.write("\n")


def read_audit_events(limit: int = 50) -> list[dict[str, Any]]:
    path = audit_log_path()
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(AuditEventContract.model_validate_json(line).model_dump())
        except Exception:
            continue
    return list(reversed(rows[-limit:]))


def candidate_key_for_row(row: pd.Series) -> str:
    parts = [
        str(row.get("patient_id", "")),
        str(row.get("hla_allele", "")),
        str(row.get("gene", row.get("gene_name", ""))),
        str(row.get("mutant_peptide", "")),
        str(row.get("protein_change", row.get("source_hgvsp_short", ""))),
    ]
    return "|".join(parts)


def _float_or(row: pd.Series, key: str, default: float = 0.0) -> float:
    val = row.get(key)
    try:
        if val is None or pd.isna(val):
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def _expression_norm(tpm: float, cap: float) -> float:
    import math

    t = max(float(tpm), 0.0)
    return max(0.0, min(1.0, math.log1p(t) / math.log1p(max(cap, 1.0))))


def _blend_expression(row: pd.Series, strategy: ResolvedStrategy) -> float:
    stub_tpm = _float_or(row, "expression_tpm", 0.0)
    n_stub = _expression_norm(stub_tpm, strategy.expression_tpm_cap)
    real_tpm = row.get("real_expression_tpm")
    if real_tpm is None or pd.isna(real_tpm):
        return n_stub
    n_real = _expression_norm(float(real_tpm), strategy.expression_tpm_cap)
    return max(0.0, min(1.0, strategy.blend_expression * n_real + (1.0 - strategy.blend_expression) * n_stub))


def _blend_ccf(row: pd.Series, strategy: ResolvedStrategy) -> float:
    stub = max(0.0, min(1.0, _float_or(row, "ccf", 0.0)))
    real_ccf = row.get("real_ccf")
    if real_ccf is None or pd.isna(real_ccf):
        return stub
    return max(0.0, min(1.0, strategy.blend_ccf * float(real_ccf) + (1.0 - strategy.blend_ccf) * stub))


def score_candidates_for_strategy(df: pd.DataFrame, strategy: ResolvedStrategy) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        for col in (
            "strategy_id",
            "strategy_origin",
            "consensus_score",
            "consensus_agreement_count",
            "coherence_score",
            "coherence_reasons",
        ):
            if col not in out.columns:
                out[col] = []
        return out

    expr_vals: list[float] = []
    ccf_vals: list[float] = []
    rl_vals: list[float] = []
    tiers: list[int] = []
    w_expr = max(0.0, float(strategy.weights.get("expression_norm", 0.0)))
    w_pres = max(0.0, float(strategy.weights.get("presentation", 0.0)))
    w_ccf = max(0.0, float(strategy.weights.get("ccf", 0.0)))
    w_sd = max(0.0, float(strategy.weights.get("self_dissimilarity", 0.0)))
    w_sum = max(w_expr + w_pres + w_ccf + w_sd, 1e-12)
    w_expr, w_pres, w_ccf, w_sd = (
        w_expr / w_sum,
        w_pres / w_sum,
        w_ccf / w_sum,
        w_sd / w_sum,
    )
    resistance_weight = max(0.0, min(1.0, abs(float(strategy.escape_penalty_weight))))
    for _, row in out.iterrows():
        expr = _blend_expression(row, strategy)
        ccf = _blend_ccf(row, strategy)
        present = max(0.0, min(1.0, _float_or(row, "presentation_score", 0.0)))
        sd = max(0.0, min(1.0, _float_or(row, "self_dissimilarity", 0.0)))
        esc = max(0.0, min(1.0, _float_or(row, "escape_penalty", 0.0)))
        immunogenicity_blend = (
            w_expr * expr + w_pres * present + w_ccf * ccf + w_sd * sd
        )
        resistance_penalty = max(0.0, min(1.0, esc * resistance_weight))
        score = immunogenicity_blend * (1.0 - resistance_penalty)
        score = max(0.0, min(1.0, score))
        if score > strategy.tier1_above:
            tier = 1
        elif score >= strategy.tier2_above:
            tier = 2
        else:
            tier = 3
        expr_vals.append(expr)
        ccf_vals.append(ccf)
        rl_vals.append(score)
        tiers.append(tier)
    out["expression_norm"] = expr_vals
    out["strategy_ccf"] = ccf_vals
    out["rl_priority"] = rl_vals
    out["tier"] = tiers
    out["strategy_id"] = strategy.strategy_id
    out["strategy_origin"] = strategy.origin
    out["scoring_profile"] = strategy.scoring_profile_id
    out["scoring_version"] = strategy.scoring_version
    out["rule_profile"] = strategy.rule_profile_id
    out["candidate_key"] = out.apply(candidate_key_for_row, axis=1)
    return attach_coherence(out, strategy)


def attach_coherence(df: pd.DataFrame, strategy: ResolvedStrategy) -> pd.DataFrame:
    out = df.copy()
    scores: list[float] = []
    reasons_blob: list[list[str]] = []
    for _, row in out.iterrows():
        penalty = 0.0
        reasons: list[str] = []
        loh = str(row.get("hla_loh_status", "")).lower()
        if loh == "lost":
            penalty += 0.35
            reasons.append("LOH loss raises escape risk.")
        elif loh == "uncertain":
            penalty += 0.15
            reasons.append("LOH status is uncertain.")

        esc = max(0.0, min(1.0, _float_or(row, "escape_penalty", 0.0)))
        if esc > 0.4:
            penalty += min(0.25, 0.35 * esc)
            reasons.append("Escape penalty is elevated.")

        expr_tpm = row.get("real_expression_tpm")
        if expr_tpm is None or pd.isna(expr_tpm):
            expr_tpm = _float_or(row, "expression_tpm", 0.0)
        try:
            expr_tpm = float(expr_tpm)
        except (TypeError, ValueError):
            expr_tpm = 0.0
        if expr_tpm < 1.0:
            penalty += 0.25
            reasons.append("Expression is near absent.")
        elif expr_tpm < 5.0:
            penalty += 0.12
            reasons.append("Expression is weak.")

        ccf = row.get("real_ccf")
        if ccf is None or pd.isna(ccf):
            ccf = _float_or(row, "ccf", 0.0)
        try:
            ccf = float(ccf)
        except (TypeError, ValueError):
            ccf = 0.0
        if ccf < 0.2:
            penalty += 0.25
            reasons.append("Candidate appears subclonal/unstable.")
        elif ccf < 0.4:
            penalty += 0.10
            reasons.append("Candidate has modest clonality support.")

        exclusions = row.get("exclusion_reasons") or []
        if isinstance(exclusions, str):
            exclusions = [exclusions] if exclusions else []
        if exclusions:
            penalty += min(0.2, 0.04 * len(exclusions))
            reasons.append("Existing exclusion flags reduce coherence.")

        score = max(0.0, min(1.0, 1.0 - penalty))
        result = CoherenceResultContract(
            candidate_key=str(row.get("candidate_key") or candidate_key_for_row(row)),
            strategy_id=strategy.strategy_id,
            coherence_score=score,
            coherence_reasons=reasons,
        )
        scores.append(result.coherence_score)
        reasons_blob.append(result.coherence_reasons)
    out["coherence_score"] = scores
    out["coherence_reasons"] = reasons_blob
    return out


def summarize_strategy_run(df: pd.DataFrame, strategy: ResolvedStrategy, *, top_n: int = 25, reference_keys: set[str] | None = None) -> dict[str, Any]:
    if df.empty:
        return {
            "strategy_id": strategy.strategy_id,
            "display_name": strategy.display_name,
            "origin": strategy.origin,
            "tier1_candidates": 0,
            "tier2_candidates": 0,
            "tier3_candidates": 0,
            "mean_rl_priority": 0.0,
            "top_rl_priority": 0.0,
            "mean_coherence": 0.0,
            "overlap_with_active": 0,
            "divergence_pct": 0.0,
            "consensus_mean": 0.0,
        }
    top = df.sort_values("rl_priority", ascending=False).head(top_n)
    top_keys = set(top["candidate_key"].astype(str).tolist())
    overlap = len(top_keys.intersection(reference_keys or top_keys))
    denom = max(len(top_keys), 1)
    return {
        "strategy_id": strategy.strategy_id,
        "display_name": strategy.display_name,
        "origin": strategy.origin,
        "tier1_candidates": int((df["tier"] == 1).sum()),
        "tier2_candidates": int((df["tier"] == 2).sum()),
        "tier3_candidates": int((df["tier"] == 3).sum()),
        "mean_rl_priority": round(float(df["rl_priority"].mean()), 4),
        "top_rl_priority": round(float(df["rl_priority"].max()), 4),
        "mean_coherence": round(float(df["coherence_score"].mean()), 4),
        "overlap_with_active": int(overlap),
        "divergence_pct": round(100.0 * (1.0 - (overlap / denom)), 1),
    }


def build_consensus_table(strategy_frames: dict[str, pd.DataFrame], *, top_n: int = 25) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n_strats = max(len(strategy_frames), 1)
    for sid, frame in strategy_frames.items():
        ranked = frame.sort_values("rl_priority", ascending=False).head(top_n).reset_index(drop=True)
        for rank_idx, (_, row) in enumerate(ranked.iterrows(), start=1):
            rows.append(
                {
                    "strategy_id": sid,
                    "candidate_key": str(row["candidate_key"]),
                    "patient_id": row.get("patient_id"),
                    "hla_allele": row.get("hla_allele"),
                    "gene": row.get("gene", row.get("gene_name")),
                    "mutant_peptide": row.get("mutant_peptide"),
                    "rl_priority": float(row.get("rl_priority", 0.0)),
                    "coherence_score": float(row.get("coherence_score", 0.0)),
                    "rank_position": rank_idx,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "candidate_key",
                "patient_id",
                "hla_allele",
                "gene",
                "mutant_peptide",
                "consensus_score",
                "consensus_agreement_count",
                "rank_dispersion",
                "mean_rl_priority",
                "mean_coherence",
            ]
        )
    raw = pd.DataFrame(rows)
    grouped = raw.groupby("candidate_key", sort=False)
    cons = grouped.agg(
        patient_id=("patient_id", "first"),
        hla_allele=("hla_allele", "first"),
        gene=("gene", "first"),
        mutant_peptide=("mutant_peptide", "first"),
        consensus_agreement_count=("strategy_id", "nunique"),
        mean_rl_priority=("rl_priority", "mean"),
        mean_coherence=("coherence_score", "mean"),
        mean_rank=("rank_position", "mean"),
        rank_dispersion=("rank_position", "std"),
    ).reset_index()
    cons["rank_dispersion"] = cons["rank_dispersion"].fillna(0.0)
    cons["consensus_score"] = (
        0.5 * (cons["consensus_agreement_count"] / n_strats)
        + 0.3 * cons["mean_rl_priority"].clip(0, 1)
        + 0.2 * cons["mean_coherence"].clip(0, 1)
    ).clip(0, 1)
    cons = cons.sort_values(
        ["consensus_score", "consensus_agreement_count", "mean_rl_priority"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return cons


class StrategyRegistry:
    list_strategies = staticmethod(list_strategies)
    get_strategy = staticmethod(get_strategy)
    clone_strategy = staticmethod(clone_strategy)
    save_strategy = staticmethod(save_strategy)
    score_candidates_for_strategy = staticmethod(score_candidates_for_strategy)
    build_consensus_table = staticmethod(build_consensus_table)
