from abc import ABC, abstractmethod
from typing import List, Literal
import pandas as pd


class ValidationResult:
    def __init__(self, valid: bool, missing_cols: List[str] = None, errors: List[str] = None):
        self.valid = valid
        self.missing_cols = missing_cols or []
        self.errors = errors or []


class BaseModule(ABC):
    NAME: str = ""
    VERSION: str = "1.0.0"
    INPUT_COLUMNS: List[str] = []
    OUTPUT_COLUMNS: List[str] = []

    def validate_inputs(self, df: pd.DataFrame) -> ValidationResult:
        missing = [c for c in self.INPUT_COLUMNS if c not in df.columns]
        return ValidationResult(valid=len(missing) == 0, missing_cols=missing)

    @abstractmethod
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the module. Must return df with OUTPUT_COLUMNS added."""
        pass

    def get_confidence_state(self, df: pd.DataFrame) -> Literal["HIGH", "MEDIUM", "LOW", "UNAVAILABLE"]:
        result = self.validate_inputs(df)
        if not result.valid:
            return "UNAVAILABLE"
        return "HIGH"

    def degrade_gracefully(self, df: pd.DataFrame, missing: List[str]) -> pd.DataFrame:
        """Called when inputs are missing. Set output cols to None."""
        for col in self.OUTPUT_COLUMNS:
            if col not in df.columns:
                df[col] = None
        # Set confidence columns to UNAVAILABLE
        confidence_col = self.NAME.lower() + "_confidence"
        if confidence_col in self.OUTPUT_COLUMNS:
            df[confidence_col] = "UNAVAILABLE"
        return df

    def safe_run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper that handles missing inputs gracefully."""
        result = self.validate_inputs(df)
        if not result.valid:
            return self.degrade_gracefully(df, result.missing_cols)
        try:
            return self.run(df)
        except Exception as e:
            print(f"Module {self.NAME} failed: {e}")
            return self.degrade_gracefully(df, [])
