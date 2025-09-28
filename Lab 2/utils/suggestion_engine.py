from typing import Dict, List, Optional
import pandas as pd
from interfaces.suggestion_interface import SuggestionEngineBase

class SuggestionEngine(SuggestionEngineBase):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def check_any(self, pref: Optional[str], column: str) -> Optional[List[str]]:
        """
        Return a list of values to filter on for `column`.
        - If the column is not present in the df, return None (ignore this column).
        - If pref is 'any' or None, return all non-null unique values from the column.
        - Otherwise return a single-item list [pref].
        """
        if column not in self.df.columns:
            return None
        if pref == 'any' or pref is None:
            return self.df[column].dropna().unique().tolist()
        return [pref]

    def get_suggestions(self, preferences: Dict[str, str]) -> pd.DataFrame:
        """
        Build a mask starting True and apply filters for each preference key
        that corresponds to a column in the dataframe.
        This allows dynamic support for additional preference columns
        such as food_quality, crowdedness, length_of_stay, etc.
        """
        # Start with all rows included
        mask = pd.Series(True, index=self.df.index)

        # Apply filters for each preference that corresponds to a dataframe column
        for slot, pref in preferences.items():
            # slot is expected to be the column name (e.g., 'food', 'area', 'food_quality', ...)
            vals = self.check_any(pref, slot)
            if vals is None:
                # column not present -> ignore this preference
                continue
            # apply filter
            mask &= self.df[slot].isin(vals)

        suggestions = self.df.loc[mask]
        return suggestions.reset_index(drop=True)
