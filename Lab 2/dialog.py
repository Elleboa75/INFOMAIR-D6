from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections.abc import Iterable
import Levenshtein
import pandas as pd
import re
import baseline 

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
        pattern = '|'.join([rf"\\b{word}\\b" for word in self.user_input_split])
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
        df_sorted = df.sort_values(by=['levenshtein_distance'])
        best_row = df_sorted.iloc[0]
        return best_row['possible_match'], best_row['levenshtein_distance']

    def match(self) -> MatchResult:
        exact_matches = self.find_exact_matches()
        if not exact_matches.empty:
            return MatchResult(matches=exact_matches, match_type="exact")
        
        fuzzy_matches_df = self.find_fuzzy_matches()
        best_match, distance = self.get_best_fuzzy_match(fuzzy_matches_df)
        return MatchResult(matches=fuzzy_matches_df, match_type="fuzzy", best_match=best_match, distance=distance)

class DialogManager:
    def __init__(self, restaurant_data_path = "restaurant_info.csv"):
        self.df = pd.read_csv(restaurant_data_path)
        self.current_state = "init"
        self.preferences = {"area": None, "food": None, "pricerange": None}
        self.text_columns = ['food', 'pricerange', 'area']
        self.templates = {
            "welcome": "Welcome, what type of restaurant are you looking for!",
            "ask_area": "What area would you perfer?",
            "ask_food": "What kind of food do you want?", 
            "ask_pricerange": "What price range?",
            "confirm_correction": "I did not recognize '{given}'. Did you mean '{corrected}'?",
            "no_match_error": "I couldn't find a match for '{value}'. Could you please rephrase?",
            "contact_info": "Here's the contact information for {restaurant}: Phone: {phone}, Address: {address}",
            "suggest": "Based on your preferences, I found {count} restaurants. Here are some options: {restaurants}",
            "goodbye": "Thank you and goodbye!"
        }
        self.suggest_counter = 0
        self.last_suggested = None
        
    def classify_dialog_act(self, utterance):
        # Classify the dialog act using the baseline model --> should be change to the NN model
        utterance = utterance.lower().strip()
        model = baseline.Train_Baseline_2()
        return model.predict_label(utterance)
    
    
    def extract_preferences(self, utterance):
        # Extract the preferences of the user using RegEx
        utterance = utterance.lower()
        extracted = {}
        food_patterns = [
            (r"(\w+)\s+food", 1), 
            (r"a\s+(?!(?:cheap|expensive|moderately)\b)(\w+)\s+restaurant", 1),  
            (r"(\w+)\s+cuisine", 1), 
            (r"serving\s+(\w+)", 1),  
        ]
        
        area_patterns = [
            (r"in\s+the\s+(\w+)", 1),
            (r"(\w+)\s+part\s+of\s+town", 1),
            (r"(\w+)\s+area", 1)
        ]
        
        price_patterns = [
            (r"(cheap|expensive|moderate)", 0), 
            (r"(moderately)\s+priced", 0),  
        ]
        
        for pattern, group in food_patterns:
            match = re.search(pattern, utterance)
            if match:
                food_word = match.group(group)
                extracted['food'] = food_word
                
        for pattern, group in area_patterns:
            match = re.search(pattern, utterance)
            if match:
                area_word = match.group(group)
                extracted['area'] = area_word
                
        for pattern, group in price_patterns:
            match = re.search(pattern, utterance)
            if match:
                price_word = match.group(group)
                extracted['pricerange'] = price_word
    
        return extracted
    
    def check_preference_validity(self, preference):
        # Check the validity of the preferences (are they in the dataset)
        for key in preference.keys():
            if not self.df[key].str.contains(rf"\b{preference[key]}\b", case=False, na=False).any() and  preference[key] != 'any':
                return (key, preference[key])
        return True
    
    def state_transition(self, current_state, user_utterance):
        # State transition flow:
            # Checks what state the flow is in --> acts on dialog_act of user
                # Extracts preferences
                # Checks validity
                # If none found find best match using Levenshtein distance
                # Swap state, print system message
        utterance = user_utterance.lower()
        dialog_act = self.classify_dialog_act(utterance)
        print(dialog_act)
        if current_state == "init":
            if dialog_act == "hello":
                return "init", self.templates["welcome"]
            
            elif dialog_act == "inform":
                extracted = self.extract_preferences(utterance) 
                validity = self.check_preference_validity(extracted)
                if validity == True:
                    self.preferences.update(extracted)
                else:
                    return self.next_missing_state(), f"What area do you want the restaurant to be located in"
                return self.next_missing_state()
            
            else:
                return "init", self.templates["welcome"]
                
        elif current_state == "ask_area":
            if dialog_act == "inform":
                if 'any' in utterance.split():
                    self.preferences['area'] = 'any'
                    return self.next_missing_state()
                else:
                    extracted = self.extract_preferences(utterance)
                    if extracted == {}:
                        extracted['area'] = RestaurantMatcher(self.df, self.text_columns, user_utterance).match().best_match
                    validity = self.check_preference_validity(extracted)
                    if validity == True:
                        self.preferences['area'] = extracted['area']
                        return self.next_missing_state()
                    else:
                        return "ask_area", self.templates["no_match_error"].format(value=validity[1])
            else:
                return "ask_area", self.templates["ask_area"]
                
        elif current_state == "ask_food":
            if dialog_act == "inform":
                if 'any' in utterance.split():
                    self.preferences['food'] = 'any'
                    return self.next_missing_state()
                else:
                    extracted = self.extract_preferences(utterance)
                    if extracted == {}:
                        extracted['food'] = RestaurantMatcher(self.df, self.text_columns, user_utterance).match().best_match
                    validity = self.check_preference_validity(extracted)
                    if validity == True:
                        self.preferences['food'] = extracted['food']
                        return self.next_missing_state()
                    else:
                        return "ask_food", self.templates["no_match_error"].format(value=validity[1])
            else:
                return "ask_food", self.templates["ask_food"]
                
        elif current_state == "ask_pricerange":
            if dialog_act == "inform":
                if 'any' in utterance.split():
                    self.preferences['pricerange'] = 'any'
                    return self.next_missing_state()
                else:
                    extracted = self.extract_preferences(utterance)
                    if extracted == {}:
                        extracted['pricerange'] = RestaurantMatcher(self.df, self.text_columns, user_utterance).match().best_match
                    validity = self.check_preference_validity(extracted)
                    if validity == True:
                        self.preferences['pricerange'] = extracted['pricerange']
                        return self.next_missing_state()
                    else:
                        return "ask_pricerange", self.templates["no_match_error"].format(value=validity[1])
            else:
                return "ask_pricerange", self.templates["ask_pricerange"]
                
        elif current_state == "suggest":
            if dialog_act == "request":
                if self.last_suggested:
                    return self.provide_contact_info(self.last_suggested)
            elif dialog_act == "bye" or dialog_act == "ack" or  dialog_act == "affirm" :
                return "end", self.templates["goodbye"]
            else:
                suggestions = self.get_suggestions()
                if len(suggestions) > 0 and len(suggestions) > self.suggest_counter:
                    suggest = suggestions.iloc[self.suggest_counter]['restaurantname']
                    self.last_suggested = suggest
                    self.suggest_counter += 1
                    return 'suggest', f'How about {suggest}?'
                else:    
                    return 'no_alts', "Sorry, I couldn't find any restaurants matching your preferences. Would you like to modify them?"
               
        elif current_state == "no_alts":
            if dialog_act == 'affirm' or dialog_act == 'ack':
                self.preferences = {"area": None, "food": None, "pricerange": None}
                return 'ask_area', 'Alright let me start over, what area do you prefer?'
            else:
                suggestions = self.get_suggestions()
                self.suggest_counter = 0
                if len(suggestions) > 0 and len(suggestions) > self.suggest_counter:
                    suggest = suggestions.iloc[self.suggest_counter]['restaurantname']
                    self.suggest_counter += 1
                    return 'suggest', f'How about {suggest}?'
                
        elif current_state == "end":
            return "end", self.templates["goodbye"]

        return current_state, "Could you please rephrase?"
    
    def next_missing_state(self):
        # Check the next state which is not filled in yet
        if self.preferences['area'] is None:
            return "ask_area", self.templates["ask_area"]
        
        elif self.preferences['food'] is None:
            return "ask_food", self.templates["ask_food"] 
        
        elif self.preferences['pricerange'] is None:
            return "ask_pricerange", self.templates["ask_pricerange"]
        
        else:
            suggestions = self.get_suggestions()
            if len(suggestions) > 0:
                suggest = suggestions.iloc[self.suggest_counter]['restaurantname']
                self.last_suggested = suggest
                self.suggest_counter += 1
                return 'suggest', f'How about {suggest}?'
            else:    
                return 'no_alts', "Sorry, I couldn't find any restaurants matching your preferences. Would you like to modify them?"
            
    
    def check_any(self, pref, column):
        # Check whether any preference is 'any'
        if pref == 'any':
            return self.df[column].unique() 
        return [pref]
    
    def get_suggestions(self):
        # Create suggestions
        food_pref = self.check_any(self.preferences['food'], 'food')
        area_pref = self.check_any(self.preferences['area'], 'area')
        price_pref = self.check_any(self.preferences['pricerange'], 'pricerange')
        suggestions = self.df.loc[(self.df['pricerange'].isin(price_pref)) & (self.df['area'].isin(area_pref)) & (self.df['food'].isin(food_pref))]
        return suggestions
    
    def provide_contact_info(self, restaurant_name):
        restaurant_row = self.df[self.df['restaurantname'].str.lower() == restaurant_name.lower()]
        
        if not restaurant_row.empty:
            restaurant = restaurant_row.iloc[0]
            contact_info = []
            
            if 'phone' in restaurant and pd.notna(restaurant['phone']):
                contact_info.append(f"Phone: {restaurant['phone']}")
            if 'addr' in restaurant and pd.notna(restaurant['addr']):
                contact_info.append(f"Address: {restaurant['addr']}")
              
            if contact_info:
                return "suggest", f"Here's the contact information for {restaurant_name}: {', '.join(contact_info)}"

    
    def run_dialog(self):
        # Run the dialog loop
        print(self.templates["welcome"])
        self.current_state = "init"
        while self.current_state != "end":
            user_input = input("User: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            
            next_state, system_response = self.state_transition(self.current_state, user_input)
            self.current_state = next_state
            #print(self.preferences)
            print(f"System: {system_response}")


# Create instance for interactive use
if __name__ == "__main__":
    dm = DialogManager()
    dm.run_dialog()

