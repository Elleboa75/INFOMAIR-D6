from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections.abc import Iterable
import pandas as pd

@dataclass
class MatchResult():
    """
    Class that holds information about a match result
    """
    matches: pd.DataFrame # DataFrame of matched rows
    match_type: str # suggests what type of match it was (exact, fuzzy)
    best_match: Optional[str] = None # if not exact match found, provide best fuzzy match
    distance: Optional[int] = None # Levenshtein distance score for fuzzy match
