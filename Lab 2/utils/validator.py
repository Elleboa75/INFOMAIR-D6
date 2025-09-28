import re
from typing import Dict, Any, Callable
import pandas as pd
from interfaces.validator_interface import PreferenceValidatorBase

class PreferenceValidator(PreferenceValidatorBase):
    """Validates candidate slot values against the dataset (and optional custom validators)."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.validators: Dict[str, Callable[[str], bool]] = {}

    def register_validator(self, slot: str, func: Callable[[str], bool]):
        self.validators[slot] = func

    def check_preference_validity(self, preferences: Dict[str, str]):
        for key, val in preferences.items():
            if val is None:
                continue
            if val == 'any':
                continue

            validator = self.validators.get(key)
            if validator is not None:
                try:
                    if not validator(val):
                        return (key, val)
                    else:
                        continue
                except Exception:
                    return (key, val)

            if key not in self.df.columns:
                return (key, val)

            if not self.df[key].str.contains(rf"\b{re.escape(val)}\b", case=False, na=False).any():
                return (key, val)
        return True
