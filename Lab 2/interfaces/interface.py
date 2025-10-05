# interfaces/interface.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Iterable, TYPE_CHECKING, Dict
import pandas as pd

# Only import concrete MatchResult for static type checking to avoid runtime circular import
if TYPE_CHECKING:
    from utils.match_result import MatchResult

class RestaurantMatcherBase(ABC):
    """
    Abstract base describing the public API of a restaurant matcher.
    Implementations are expected to be initialised with (df, text_columns, user_input).
    """
    df: pd.DataFrame
    text_columns: list[str]
    user_input_split: list[str]

    @abstractmethod
    def find_exact_matches(self) -> pd.DataFrame:
        """Return DataFrame rows that match the user's tokens exactly (whole-word)."""
        pass

    @abstractmethod
    def find_fuzzy_matches(self, max_distance: int = 3) -> pd.DataFrame:
        """
        Return a DataFrame of fuzzy candidate matches with columns:
        ['input_word', 'possible_match', 'levenshtein_distance'].
        """
        pass

    @abstractmethod
    def get_best_fuzzy_match(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[int]]:
        """
        Given a fuzzy-matches DataFrame, return (best_match, distance) or (None, None).
        """
        pass

    @abstractmethod
    def match(self) -> "MatchResult":
        """
        High-level convenience: return a MatchResult summarising either exact or fuzzy results.
        This is the method your DialogManager uses and therefore should be part of the interface.
        """
        pass

