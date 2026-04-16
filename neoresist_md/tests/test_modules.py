from __future__ import annotations

import pandas as pd

from neoresist_md.backend.core.generation.mhcflurry_module import MHCFlurryModule


def test_generation_module_runs():
    df = pd.DataFrame([{"patient_id": "P1", "gene": "TP53"}])
    out = MHCFlurryModule().safe_run(df)
    assert "binding_affinity" in out.columns
    assert "binding_confidence" in out.columns
