from typing import Dict
import pandas as pd
from interfaces.suggestion_interface import SuggestionEngineBase

class SuggestionEngine(SuggestionEngineBase):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def check_any(self, pref: str, column: str):
        if pref == 'any' or pref is None:
            return self.df[column].unique().tolist()
        return [pref]

    def get_suggestions(self, preferences: Dict[str, str]) -> pd.DataFrame:
        food_pref = self.check_any(preferences.get('food'), 'food')
        area_pref = self.check_any(preferences.get('area'), 'area')
        price_pref = self.check_any(preferences.get('pricerange'), 'pricerange')
        suggestions = self.df.loc[
            (self.df['pricerange'].isin(price_pref)) &
            (self.df['area'].isin(area_pref)) &
            (self.df['food'].isin(food_pref))
        ]
        return suggestions.reset_index(drop=True)
