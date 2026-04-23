"""Strategy-kind handler registry.

A "kind" plugs a new scoring approach into the NeoResist-MD strategy registry
without touching the Dash UI or the rest of the pipeline. Each handler takes
a candidate DataFrame plus a ResolvedStrategy and returns a DataFrame with
at minimum an `rl_priority` column in [0, 1]. The registry dispatcher
(neoresist.strategy_registry.score_candidates_for_strategy) fills in the
remaining UI-facing columns and tier assignment.

Kinds currently registered:
    ml_model   - load a fitted sklearn-compatible .pkl and call predict_proba
                 on a configured feature matrix.
    neoguider  - apply the aKDE -> IR -> CIR feature transform (port of
                 Zhao et al. Genome Medicine 2026) then score with the
                 packaged logistic regression head.

Add a new kind by registering a callable under KIND_HANDLERS here or via
`register_kind_handler(name, fn)` at import time elsewhere.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from neoresist.strategy_registry import ResolvedStrategy

KindHandler = Callable[[pd.DataFrame, "ResolvedStrategy"], pd.DataFrame]

KIND_HANDLERS: dict[str, KindHandler] = {}


def register_kind_handler(name: str, handler: KindHandler) -> None:
    KIND_HANDLERS[name] = handler


def get_kind_handler(name: str) -> KindHandler | None:
    if not KIND_HANDLERS:
        _load_default_handlers()
    return KIND_HANDLERS.get(name)


def _load_default_handlers() -> None:
    try:
        from neoresist.strategies.ml_model import score_ml_model
        KIND_HANDLERS.setdefault("ml_model", score_ml_model)
    except Exception:
        pass
    try:
        from neoresist.strategies.neoguider import score_neoguider
        KIND_HANDLERS.setdefault("neoguider", score_neoguider)
    except Exception:
        pass


__all__ = ["get_kind_handler", "register_kind_handler", "KIND_HANDLERS"]

# Backwards-compatible aliases used by earlier session notes.
get_handler = get_kind_handler
register_handler = register_kind_handler
