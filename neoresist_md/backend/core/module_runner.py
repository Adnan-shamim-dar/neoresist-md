import yaml
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class ModuleRunner:
    """Orchestrates module execution. Tracks state. Never reruns completed modules."""

    def __init__(self, config_path: str = "config/"):
        self.config_path = Path(config_path)
        if not self.config_path.is_dir():
            self.config_path = Path(__file__).resolve().parents[2] / "config"
        self.weights = self._load_weights()
        self.tool_registry = self._load_tool_registry()
        self.modules = self._init_modules()
        self.run_state: Dict[str, str] = {}  # module_name -> status

    def _load_weights(self) -> dict:
        with open(self.config_path / "module_weights.yaml") as f:
            return yaml.safe_load(f)

    def _load_tool_registry(self) -> dict:
        with open(self.config_path / "tool_registry.yaml") as f:
            return yaml.safe_load(f)

    def _init_modules(self) -> dict:
        """Initialise all available modules."""
        from neoresist_md.backend.core.generation.mhcflurry_module import MHCflurryModule
        from neoresist_md.backend.core.expression.tcga_expression_module import ExpressionModule
        from neoresist_md.backend.core.clonality.pyclone_module import ClonalityModule
        from neoresist_md.backend.core.recognition.foreignness_module import ForeignnessModule
        from neoresist_md.backend.core.resistance_loop.resistance_module import ResistanceLoopModule
        from neoresist_md.backend.core.prioritization.tiering_module import TieringModule
        return {
            "generation": MHCflurryModule(),
            "expression": ExpressionModule(),
            "clonality": ClonalityModule(),
            "recognition": ForeignnessModule(),
            "resistance": ResistanceLoopModule(),
            "tiering": TieringModule(),
        }

    def get_module_status(self, patient_id: str) -> Dict[str, str]:
        """Return status dict for UI display."""
        # Returns: {module_name: "COMPLETE"|"RUNNING"|"PENDING"|"UNAVAILABLE"|"ERROR"}
        statuses: Dict[str, str] = {}
        for name in self.modules.keys():
            statuses[name] = self.run_state.get(name, "PENDING")
        return statuses

    def run_pipeline(
        self,
        df: pd.DataFrame,
        enabled_modules: Optional[List[str]] = None,
        weights_override: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Run all enabled modules in sequence."""
        if enabled_modules is None:
            enabled_modules = list(self.modules.keys())
        weights = weights_override or self.weights
        for name, module in self.modules.items():
            if name not in enabled_modules:
                continue
            if self.run_state.get(name) == "COMPLETE":
                continue
            self.run_state[name] = "RUNNING"
            try:
                df = module.safe_run(df)
                conf_col = f"{module.NAME.lower()}_confidence"
                if conf_col in df.columns and df[conf_col].astype(str).str.upper().eq("UNAVAILABLE").all():
                    self.run_state[name] = "UNAVAILABLE"
                else:
                    self.run_state[name] = "COMPLETE"
            except Exception:
                self.run_state[name] = "ERROR"
                raise
        return df

    def save_strategy(self, name: str, weights: dict, tool_selections: dict, source: str = "user_defined"):
        """Save a user strategy to the strategy library."""
        strategy = {
            "strategy_id": name,
            "created_at": datetime.now().isoformat(),
            "source": source,
            "weight_vector": weights,
            "tool_selections": tool_selections,
        }
        library_path = self.config_path / "strategy_library.json"
        if library_path.exists():
            with open(library_path) as f:
                library = json.load(f)
        else:
            library = []
        if isinstance(library, dict):
            library = list(library.get("strategies") or [])
        library.append(strategy)
        with open(library_path, "w") as f:
            json.dump(library, f, indent=2)
