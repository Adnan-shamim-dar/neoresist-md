from __future__ import annotations

import pandas as pd

from neoresist.version import __version__


def stamp_enrichment_metadata(df: pd.DataFrame, *, dataset_name: str) -> pd.DataFrame:
    out = df.copy()
    out["dataset_name"] = dataset_name
    out["app_version"] = __version__
    return out
