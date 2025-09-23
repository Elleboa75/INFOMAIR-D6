# interfaces/interface.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Iterable, TYPE_CHECKING
import pandas as pd

# Only import concrete MatchResult for static type checking to avoid runtime circular import
if TYPE_CHECKING:
    from utils.match_result import MatchResult

class DialogManagerBase(ABC):
    """
    Abstract interface for a dialog manager that collects preferences and
    suggests restaurants.
    """

    @abstractmethod
    def classify_dialog_act(self, utterance: str) -> str:
        """Return a dialog act label (e.g. 'inform', 'hello', 'bye', ...)."""
        pass

    @abstractmethod
    def extract_preferences(self, utterance: str) -> dict[str, str]:
        """Extract preference key->value pairs from the utterance (may be empty)."""
        pass

    @abstractmethod
    def check_preference_validity(self, preference: dict[str, str]) -> bool | tuple[str, str]:
        """
        Return True if all provided preferences are valid (present in dataset).
        Otherwise return (key, value) indicating the invalid preference.
        """
        pass

    @abstractmethod
    def state_transition(self, current_state: str, user_utterance: str) -> tuple[str, str]:
        """Given current_state and user text, return (next_state, system_message)."""
        pass

    @abstractmethod
    def next_missing_state(self) -> tuple[str, str]:
        """Return (next_state, prompt) for the next missing preference."""
        pass

    @abstractmethod
    def check_any(self, pref: Optional[str], column: str) -> Iterable[str]:
        """
        If pref == 'any' return all possible values for `column` (iterable),
        otherwise return iterable containing the single pref.
        """
        pass

    @abstractmethod
    def get_suggestions(self) -> pd.DataFrame:
        """Return a DataFrame with matching restaurants given current preferences."""
        pass

    @abstractmethod
    def run_dialog(self) -> None:
        """Main loop (side-effecting); prints/prompts until dialog ends."""
        pass

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