from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections.abc import Iterable
import Levenshtein
import pandas as pd

@dataclass
class MatchResult:
    matches: pd.DataFrame # DataFrame of matched rows
    match_type: str # suggests what type of match it was (exact, fuzzy)
    best_match: Optional[str] = None # if not exact match found, provide best fuzzy match
    distance: Optional[int] = None # levenshtein distance score for fuzzy match

class RestaurantMatcher:
    def __init__(self, df: pd.DataFrame, text_columns: List[str], user_input: str):
        self.df = df
        self.text_columns = text_columns
        self.user_input_split = user_input.lower().split()
        
    def find_exact_matches(self) -> pd.DataFrame:
        # | is the regex OR operator
        pattern = '|'.join([rf"\b{word}\b" for word in self.user_input_split])
        mask = self.df[self.text_columns].apply(lambda col: col.str.contains(pattern, case=False, na=False)).any(axis=1)
        return self.df[mask]
    
    def find_fuzzy_matches(self, max_distance: int=3) -> pd.DataFrame:
        possible_matches = []
        # Handle main text columns
        for column in self.text_columns:
            unique_values = self.df[column].dropna().astype(str).str.lower().unique()
            possible_matches.extend(list(self._get_fuzzy_matches(self.user_input_split, unique_values, max_distance)))
        # Remove possible duplicates
        df_matches = pd.DataFrame(possible_matches, columns=["input_word", "possible_match", "levenshtein_distance"]).drop_duplicates()
        return df_matches
    
    def _get_fuzzy_matches(self, words: Iterable[str], candidates: Iterable[str], max_distance: int):
        for word in words:
            # Pre-filter candidates so only reasonably similar-length strings are compared (performance optimization)
            filtered_candidates = [c for c in candidates if abs(len(c) - len(word)) <= max_distance]
            for candidate in filtered_candidates:
                distance = Levenshtein.distance(word, str(candidate).lower())
                if distance <= max_distance:
                    yield [word, candidate, distance]
    
    def get_best_fuzzy_match(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[int]]:
        if df.empty:
            return None, None
        df.sort_values(by=['levenshtein_distance'], inplace=True)
        best_row = df.iloc[0]
        return best_row['possible_match'], best_row['levenshtein_distance']

    
    def match(self) -> MatchResult:
        exact_matches = self.find_exact_matches()
        if not exact_matches.empty:
            return MatchResult(matches=exact_matches, match_type="exact")
        
        fuzzy_matches_df = self.find_fuzzy_matches()
        best_match, distance = self.get_best_fuzzy_match(fuzzy_matches_df)
        return MatchResult(matches=fuzzy_matches_df, match_type="fuzzy", best_match=best_match, distance=distance)

if __name__ == "__main__":
    # load data
    df = pd.read_csv("assets/restaurant_info.csv")
    
    # user input
    user_input = "I want to go turkishh".lower()
    text_columns = ['food', 'pricerange', 'area']
    
    matcher = RestaurantMatcher(df, text_columns, user_input)
    result = matcher.match()
    print(result.matches)