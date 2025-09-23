from interfaces.interface import RestaurantMatcherBase
from .match_result import MatchResult  # your concrete MatchResult dataclass / class

from typing import List, Tuple, Optional, Iterable
import pandas as pd
import Levenshtein
from .match_result import MatchResult


class RestaurantMatcher(RestaurantMatcherBase):
    def __init__(self, df: pd.DataFrame, text_columns: List[str], user_input: str):
        self.df = df
        self.text_columns = text_columns
        self.user_input_split = user_input.lower().split()

    def find_exact_matches(self) -> pd.DataFrame:
        """Find exact word matches in the text columns using regex word boundaries."""
        pattern = '|'.join([rf"\b{word}\b" for word in self.user_input_split])
        mask = self.df[self.text_columns].apply(
            lambda col: col.str.contains(pattern, case=False, na=False)
        ).any(axis=1)
        return self.df[mask]

    def find_fuzzy_matches(self, max_distance: int = 3) -> pd.DataFrame:
        """Find fuzzy matches based on Levenshtein distance."""
        possible_matches = []

        for column in self.text_columns:
            unique_values = (
                self.df[column].dropna().astype(str).str.lower().unique()
            )
            possible_matches.extend(
                list(self._get_fuzzy_matches(self.user_input_split, unique_values, max_distance))
            )

        df_matches = pd.DataFrame(
            possible_matches,
            columns=["input_word", "possible_match", "levenshtein_distance"]
        ).drop_duplicates()

        return df_matches

    def _get_fuzzy_matches(
        self,
        words: Iterable[str],
        candidates: Iterable[str],
        max_distance: int
    ):
        """Yield candidate matches within max_distance Levenshtein distance."""
        for word in words:
            filtered_candidates = [
                c for c in candidates if abs(len(c) - len(word)) <= max_distance
            ]
            for candidate in filtered_candidates:
                distance = Levenshtein.distance(word, str(candidate).lower())
                if distance <= max_distance:
                    yield [word, candidate, distance]

    def get_best_fuzzy_match(
        self, df: pd.DataFrame
    ) -> Tuple[Optional[str], Optional[int]]:
        """Return the best fuzzy match (closest Levenshtein distance)."""
        if df.empty:
            return None, None
        df_sorted = df.sort_values(by=['levenshtein_distance'])
        best_row = df_sorted.iloc[0]
        return best_row['possible_match'], best_row['levenshtein_distance']

    def match(self) -> MatchResult:
        """Try exact matches first; fall back to fuzzy matching if needed."""
        exact_matches = self.find_exact_matches()
        if not exact_matches.empty:
            return MatchResult(matches=exact_matches, match_type="exact")

        fuzzy_matches_df = self.find_fuzzy_matches()
        best_match, distance = self.get_best_fuzzy_match(fuzzy_matches_df)
        return MatchResult(
            matches=fuzzy_matches_df,
            match_type="fuzzy",
            best_match=best_match,
            distance=distance,
        )