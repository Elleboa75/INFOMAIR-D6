from typing import Tuple, Dict, Any
import re
import json
import os
import time
import sys

import pandas as pd

from .restaurant_matcher import RestaurantMatcher
from .extractor import SlotExtractor
from .validator import PreferenceValidator
from .suggestion_engine import SuggestionEngine
from .parser import ChangeRequestParser
from interfaces.dialog_manager_interface import DialogManagerBase


class DialogManager(DialogManagerBase):
    """Coordinator that composes the small components (extractor, validator,
    suggestion engine, parser) and implements the dialog policy.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 config_path: str = "dialog_config.json",
                 model: Any = None,
                 extractor: SlotExtractor = None,
                 validator: PreferenceValidator = None,
                 sugg_engine: SuggestionEngine = None,
                 parser: ChangeRequestParser = None,
                 all_caps = False,
                 allow_restarts = True,
                 delay = 0,
                 formal = True):
        
        # data + config
        self.df = df
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.all_caps = all_caps
        self.allow_restart = allow_restarts
        self.delay = delay
        self.last_suggested = None

        # templates and states
        if formal:
            self.templates = self.config.get('formal_templates', {})
        else:
            self.templates = self.config.get("informal_templates", {})

        states = self.config.get('states', {})
        self.slot_states = states.get('slot_states', {})
        # --- NEW: automatically add extra slot states if the df contains those columns
        # minimal additions: add ask_food_quality, ask_crowdedness, ask_length_of_stay
        extra_slots = [
            ('ask_food_quality', 'foodquality'),
            ('ask_crowdedness', 'crowdedness'),
            ('ask_length_of_stay', 'lengthstay'),
        ]
        for state_name, slot_name in extra_slots:
            if slot_name in self.df.columns and slot_name not in self.slot_states.values():
                # add to slot_states so dialog will prompt for them
                self.slot_states[state_name] = slot_name

        # separate basic slots from extra slots
        self.basic_slots = ['ask_area', 'ask_food', 'ask_pricerange']
        self.extra_slots = [s for s in self.slot_states.keys() if s not in self.basic_slots]

        # slot order: basic slots first, then extra slots
        self.slot_order = [s for s in self.basic_slots if s in self.slot_states] + self.extra_slots
        # --- END NEW

        self.suggest_state = states.get('suggest_state', 'suggest')
        self.no_alts_state = states.get('no_alts_state', 'no_alts')
        self.additional_req_state = states.get('additional_req_state', 'add_req')
        self.end_state = states.get('end_state', 'end')
        self.init_state = states.get('init_state', 'init')

        # dialog bookkeeping
        self.current_state = self.init_state
        # ensure preferences always has the expected keys (dynamically include any new slots)
        self.preferences = {v: None for v in self.slot_states.values()}
        self.text_columns = list(self.slot_states.values())
        self.suggest_counter = 0
        self.additional_req_asked = False  # Track if we've asked for additional requirements

        # components (allow dependency injection for testing)
        self.extractor = extractor or SlotExtractor.from_config_file(config_path)
        self.validator = validator or PreferenceValidator(self.df)
        self.sugg_engine = sugg_engine or SuggestionEngine(self.df)
        self.parser = parser or ChangeRequestParser()

        # model for dialog act classification (optional)
        self.model = model
        if self.model is None:
            try:
                import baseline
                self.model = baseline.Train_Baseline_2()
            except Exception:
                self.model = None

    def _load_config(self, path: str) -> Dict[str, Any]:
        if path and os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)

        return {
            "templates": {
    "welcome": "Hello , welcome to the Cambridge restaurant system? You can ask for restaurants by area , price range or food type . How may I help you?",
    "ask_area": "What area would you prefer?",
    "ask_food": "What kind of food do you want?",
    "ask_pricerange": "What price range?",
    "confirm_correction": "I did not recognize '{given}'. Did you mean '{corrected}'?",
    "no_match_error": "I couldn't find a match for '{value}'. Could you please rephrase?",
    "suggest": "Based on your preferences, I found {count} restaurants. Here are some options: {restaurants}",
    "goodbye": "Thank you and goodbye!",
    "no_alts": "Sorry, I couldn't find any restaurants matching your preferences. Would you like to modify them?"
  },
            "states": {
    "slot_states": {
      "ask_area": "area",
      "ask_food": "food",
      "ask_pricerange": "pricerange"
    },
    "suggest_state": "suggest",
    "no_alts_state": "no_alts",
    "end_state": "end",
    "init_state": "init"
        }
        }
    def classify_dialog_act(self, utterance: str) -> str:
        u = (utterance or '').lower().strip()
        if self.model is None:
            if any(w in u for w in ["hello", "hi", "hey"]):
                return "hello"
            if any(w in u for w in ["bye", "goodbye"]):
                return "bye"
            return "inform"
        try:
            res = self.model.predict(u)
            print(res)
            if isinstance(res, dict) and 'predicted_dialog_act' in res:
                return res['predicted_dialog_act']
            if hasattr(self.model, 'predict_label'):
                return self.model.predict_label(u)
            if isinstance(res, str):
                return res
        except Exception:
            pass
        return "inform"

    def next_missing_state(self) -> Tuple[str, str]:
        # Check if basic slots (area, food, pricerange) are all filled
        basic_slots_filled = all(
            self.preferences.get(self.slot_states.get(state)) is not None
            for state in self.basic_slots
            if state in self.slot_states
        )

        # If basic slots are filled and we have extra slots available and haven't asked yet
        if basic_slots_filled and self.extra_slots and not self.additional_req_asked:
            return self.additional_req_state, self.templates.get('additional_req', "Do you have any additional requirements?")

        # Check for missing slots in order (basic first, then extra)
        for state in self.slot_order:
            slot = self.slot_states.get(state)
            if slot is None:
                continue
            if self.preferences.get(slot) is None:
                return state, self.templates.get(state, self.templates.get(f"ask_{slot}", ""))

        suggestions = self.sugg_engine.get_suggestions(self.preferences)
        if len(suggestions) > 0:
            suggestion_name = suggestions.iloc[self.suggest_counter % len(suggestions)]['restaurantname']
            self.last_suggested = suggestion_name
            self.suggest_counter += 1
            return self.suggest_state, f"How about {suggestion_name}?"
        return self.no_alts_state, self.templates.get('no_alts', "I couldn't find matches")

    def _handle_slot_state(self, state: str, utterance: str, dialog_act: str) -> Tuple[str, str]:
        slot = self.slot_states[state]
        if dialog_act in ['inform', 'null']: ## added because the dialog act makes mistakes
            if 'any' in (utterance or '').split():
                self.preferences[slot] = 'any'
                return self.next_missing_state()

            extracted = self.extractor.extract_preferences(utterance)
            # slot-specific single-token fallback (when system is explicitly asking a slot)
            if not extracted or slot not in extracted:
                tokens = (utterance or '').strip().split()
                if len(tokens) == 1 and tokens[0]:
                    extracted[slot] = tokens[0].lower()
                else:
                    try:
                        match = RestaurantMatcher(self.df, self.text_columns, utterance).match()
                        best = getattr(match, 'best_match', None)
                        if best:
                            extracted[slot] = best
                    except Exception:
                        pass

            # validate only the current slot value extracted
            validity = self.validator.check_preference_validity({k: v for k, v in extracted.items() if k == slot})
            if validity is True:
                if slot in extracted:
                    self.preferences[slot] = extracted[slot]
                return self.next_missing_state()
            return state, self.templates.get('no_match_error', "I couldn't find a match").format(value=validity[1])

        return state, self.templates.get(state, self.templates.get(f"ask_{slot}", ""))

    def state_transition(self, current_state: str, user_utterance: str) -> Tuple[str, str]:
        utterance = (user_utterance or '').lower()
        dialog_act = self.classify_dialog_act(utterance)

        if current_state == self.init_state:
            if dialog_act == 'hello':
                return self.init_state, self.templates.get('welcome', 'Welcome')
            if dialog_act == 'inform':
                extracted = self.extractor.extract_preferences(utterance)

                if not extracted:
                    tokens = utterance.strip().split()
                    if len(tokens) == 1 and tokens[0]:
                        tok = tokens[0].lower()
                        # priority: food, area, pricerange (preserve existing priority),
                        # then any of the new text slots if present
                        priority_cols = [c for c in ['food', 'area', 'pricerange', 'food_quality', 'crowdedness', 'length_of_stay'] if c in self.text_columns]
                        for col in priority_cols:
                            if col in self.text_columns and self.df[col].str.contains(rf"\b{re.escape(tok)}\b", case=False, na=False).any():
                                extracted[col] = tok
                                break

                validity = self.validator.check_preference_validity(extracted)
                if validity is True:
                    self.preferences.update(extracted)
                    return self.next_missing_state()
                return self.next_missing_state()
            return self.init_state, self.templates.get('welcome', 'Welcome')

        if current_state in self.slot_states:
            return self._handle_slot_state(current_state, user_utterance, dialog_act)

        if current_state == self.additional_req_state:
            self.additional_req_asked = True 
            if dialog_act in ['affirm', 'ack']:
                return self.next_missing_state()
            elif dialog_act in ['negate', 'deny']:
                # Fill all extra slots with 'any' to skip them
                for state in self.extra_slots:
                    slot = self.slot_states.get(state)
                    if slot and self.preferences.get(slot) is None:
                        self.preferences[slot] = 'any'

                suggestions = self.sugg_engine.get_suggestions(self.preferences)
                if len(suggestions) > 0:
                    suggestion_name = suggestions.iloc[self.suggest_counter % len(suggestions)]['restaurantname']
                    self.last_suggested = suggestion_name
                    self.suggest_counter += 1
                    return self.suggest_state, f"How about {suggestion_name}?"
                return self.no_alts_state, self.templates.get('no_alts', "I couldn't find matches")
            else:
                self.additional_req_asked = False
                return self.additional_req_state, self.templates.get('additional_req', "Do you have any additional requirements?")

        if current_state == self.suggest_state:
            print(dialog_act)
            if dialog_act == "request":
                if self.last_suggested:
                    return self.provide_contact_info(self.last_suggested)
                
            if dialog_act in ['bye', 'ack', 'affirm']:
                return self.end_state, self.templates.get('goodbye', 'Goodbye')
            suggestions = self.sugg_engine.get_suggestions(self.preferences)

            if len(suggestions) > 0 and self.suggest_counter < len(suggestions):
                suggestion_name = suggestions.iloc[self.suggest_counter]['restaurantname']
                self.last_suggested = suggestion_name
                self.suggest_counter += 1
                return self.suggest_state, f"How about {suggestion_name}?"

            return self.no_alts_state, self.templates.get('ask_change_slot', 'No alternatives')

        if current_state == self.no_alts_state:
            if dialog_act in ['affirm', 'ack']:
                self.preferences = {v: None for v in self.slot_states.values()}
                self.suggest_counter = 0
                self.additional_req_asked = False
                return self.next_missing_state()

            slot, value = self.parser.parse_change_request(utterance)
            if slot == 'restart':
                if self.allow_restart == True:
                    self.preferences = {v: None for v in self.slot_states.values()}
                    self.suggest_counter = 0
                    self.additional_req_asked = False
                    return self.next_missing_state()[0], self.templates.get('reset_confirm', 'Restarting') + " " + self.templates.get(self.next_missing_state()[0], '')
                else:
                    return self.no_alts_state, self.templates.get('no_reset','No resets allowed')


            if slot is None and value == 'any':
                # --- UPDATED: iterate over all text slots, not only area/food/pricerange
                for s in self.text_columns:
                    if self.preferences.get(s) not in [None, 'any']:
                        self.preferences[s] = 'any'
                        self.suggest_counter = 0
                        return self.next_missing_state()
                return self.no_alts_state, self.templates.get('ask_change_slot', 'Which slot to change?')

            # --- UPDATED: allow changing any recognized slot (not just area/food/pricerange)
            if slot in self.text_columns:
                if value is None:
                    self.preferences[slot] = None
                    self.suggest_counter = 0
                    prompt_slot = 'price range' if slot == 'pricerange' else slot
                    state_name = next((s for s, sl in self.slot_states.items() if sl == slot), None)
                    return state_name, self.templates.get('confirm_change', "Okay—let's update {slot}.").format(slot=prompt_slot)
                else:
                    if value == 'any':
                        self.preferences[slot] = 'any'
                        self.suggest_counter = 0
                        return self.next_missing_state()
                    is_valid = self.validator.check_preference_validity({slot: value})
                    if is_valid is True:
                        self.preferences[slot] = value
                        self.suggest_counter = 0
                        return self.next_missing_state()
                    return self.no_alts_state, self.templates.get('no_match_error', 'No match').format(value=value)

            return self.no_alts_state, self.templates.get('ask_change_slot', 'Which would you like to change?')

        if current_state == self.end_state:
            return self.end_state, self.templates.get('goodbye', 'Goodbye')

        return current_state, 'Could you please rephrase?'
    
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
            
    def show_thinking(self):
        print("Thinking...", end="", flush=True)
        time.sleep(self.delay)
        print("\r" + " " * 12 + "\r", end="", flush=True)
        
    def run_dialog(self):
        if self.all_caps:
            print(self.templates.get('welcome', 'Welcome').upper())
        else:
            print(self.templates.get('welcome', 'Welcome'))

        self.current_state = self.init_state
        while self.current_state != self.end_state:
            user_input = input('User: ')
            if user_input.lower() in ['quit', 'exit']:
                break
            next_state, system_response = self.state_transition(self.current_state, user_input)
            print(self.preferences)
            print(next_state)
            self.current_state = next_state
            if self.all_caps:
                self.show_thinking()
                print(f"System: {system_response}".upper())
            else:
                self.show_thinking()
                print(f"System: {system_response}")
