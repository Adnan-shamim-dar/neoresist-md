from __future__ import annotations

import math

import pandas as pd

EXPRESSION_TPM_CAP = 1000.0

# When real TPM / CCF exist, blend toward real evidence (0 = stub only, 1 = real only).
W_BLEND_EXPRESSION_REAL = 0.85
W_BLEND_CCF_REAL = 0.85


def expression_norm_from_tpm(tpm: float) -> float:
    """Map TPM to ``[0, 1]`` with a monotone log curve (cap at ``EXPRESSION_TPM_CAP``)."""
    t = max(float(tpm), 0.0)
    return max(0.0, min(1.0, math.log1p(t) / math.log1p(EXPRESSION_TPM_CAP)))


def _self_dissimilarity_fill(peptide: str, gene: str, existing: object) -> float:
    if existing is not None and pd.notna(existing):
        try:
            v = float(existing)
            if not math.isnan(v):
                return max(0.0, min(1.0, v))
        except (TypeError, ValueError):
            pass
    h = hash((gene or "", peptide or "")) % (2**32)
    return h / (2**32 - 1)


def tier_from_rl(rl: float) -> int:
    """Tier 1 if RL > 0.7; tier 2 if ``0.4 <= RL <= 0.7``; tier 3 if RL < 0.4."""
    if rl > 0.7:
        return 1
    if rl >= 0.4:
        return 2
    return 3


def _float_or(row: pd.Series, key: str, default: float = 0.0) -> float:
    v = row.get(key)
    try:
        if v is None or pd.isna(v):
            return default
        x = float(v)
        if math.isnan(x):
            return default
        return x
    except (TypeError, ValueError):
        return default


def _expression_norm_for_row(row: pd.Series, *, prefer_real: bool) -> tuple[float, str]:
    """Return (blended expression norm, source tag)."""
    stub_tpm = _float_or(row, "expression_tpm", 0.0)
    n_stub = expression_norm_from_tpm(stub_tpm)
    if not prefer_real:
        return n_stub, "stub"

    rt = row.get("real_expression_tpm")
    if rt is None or pd.isna(rt):
        return n_stub, "stub"
    n_real = expression_norm_from_tpm(float(rt))
    w = W_BLEND_EXPRESSION_REAL
    blend = w * n_real + (1.0 - w) * n_stub
    return max(0.0, min(1.0, blend)), "blended_real"


def _ccf_for_row(row: pd.Series, *, prefer_real: bool) -> tuple[float, str]:
    stub_c = max(0.0, min(1.0, _float_or(row, "ccf", 0.0)))
    if not prefer_real:
        return stub_c, "stub"

    rc = row.get("real_ccf")
    if rc is None or pd.isna(rc):
        return stub_c, "stub"
    n_real = max(0.0, min(1.0, float(rc)))
    w = W_BLEND_CCF_REAL
    blend = w * n_real + (1.0 - w) * stub_c
    return max(0.0, min(1.0, blend)), "blended_real"


def apply_resistance_loop(df: pd.DataFrame, *, prefer_real_evidence: bool = False) -> pd.DataFrame:
    """
    Compute ``rl_priority``, ``tier``, and ensure ``self_dissimilarity`` is populated.

    When ``prefer_real_evidence`` is True and ``real_expression_tpm`` / ``real_ccf`` exist,
    blends toward real measurements (see module constants).

    ``rl_priority = 0.2*expression_norm + 0.3*presentation + 0.3*ccf + 0.1*self_dissimilarity
    - 0.2*escape_penalty``
    """
    out = df.copy()
    if out.empty:
        return out

    use_real = bool(prefer_real_evidence)
    if use_real and "real_expression_tpm" not in out.columns and "real_ccf" not in out.columns:
        use_real = False

    if "escape_penalty" not in out.columns:
        out["escape_penalty"] = 0.0

    expr_src: list[str] = []
    ccf_src: list[str] = []
    expr_norm: list[float] = []
    pres: list[float] = []
    ccf: list[float] = []
    sd: list[float] = []
    esc: list[float] = []

    for _, row in out.iterrows():
        en, esrc = _expression_norm_for_row(row, prefer_real=use_real)
        expr_norm.append(en)
        expr_src.append(esrc)

        ccf_b, csrc = _ccf_for_row(row, prefer_real=use_real)
        ccf.append(ccf_b)
        ccf_src.append(csrc)

        ps_f = max(0.0, min(1.0, _float_or(row, "presentation_score", 0.0)))
        pres.append(ps_f)

        pep = str(row.get("mutant_peptide", ""))
        gen = str(row.get("gene", row.get("gene_name", "")))
        ex_sd = row.get("self_dissimilarity")
        sd.append(_self_dissimilarity_fill(pep, gen, ex_sd))

        esc.append(max(0.0, min(1.0, _float_or(row, "escape_penalty", 0.0))))

    out["expression_norm"] = expr_norm
    out["evidence_expression_source"] = expr_src
    out["evidence_ccf_source"] = ccf_src
    out["self_dissimilarity"] = sd

    rl = []
    for e_n, p, c, s, e_p in zip(expr_norm, pres, ccf, sd, esc, strict=True):
        score = 0.2 * e_n + 0.3 * p + 0.3 * c + 0.1 * s - 0.2 * e_p
        rl.append(max(0.0, min(1.0, score)))
    out["rl_priority"] = rl
    out["tier"] = [tier_from_rl(x) for x in rl]

    return out
