from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd

class SuggestionEngineBase(ABC):
    @abstractmethod
    def get_suggestions(self, preferences: Dict[str, str]) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def check_any(self, pref: str, column: str):
        raise NotImplementedError
