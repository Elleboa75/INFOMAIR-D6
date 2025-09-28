from typing import Tuple, Dict, Any
import re
import json
import os

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
                 parser: ChangeRequestParser = None):
        # data + config
        self.df = df
        self.config_path = config_path
        self.config = self._load_config(config_path)

        # templates and states
        self.templates = self.config.get('templates', {})
        states = self.config.get('states', {})
        self.slot_states = states.get('slot_states', {})
        # --- NEW: automatically add extra slot states if the df contains those columns
        # minimal additions: add ask_food_quality, ask_crowdedness, ask_length_of_stay
        extra_slots = [
            ('ask_food_quality', 'food_quality'),
            ('ask_crowdedness', 'crowdedness'),
            ('ask_length_of_stay', 'length_of_stay'),
        ]
        for state_name, slot_name in extra_slots:
            if slot_name in self.df.columns and slot_name not in self.slot_states.values():
                # add to slot_states so dialog will prompt for them
                self.slot_states[state_name] = slot_name
                # add simple default template if none present
                if f"ask_{slot_name}" not in self.templates:
                    # human-friendly wording for length_of_stay
                    prompt = "How long do you plan to stay?" if slot_name == 'length_of_stay' else f"What {slot_name.replace('_', ' ')} would you prefer?"
                    self.templates[f"ask_{slot_name}"] = prompt

        # original slot order based on keys of slot_states
        self.slot_order = states.get('slot_order', list(self.slot_states.keys()))
        # if slot_order from config doesn't include newly added states, recompute so new states are asked later
        # keep the original order for existing ones, then append any new ones
        config_order = states.get('slot_order', None)
        if config_order is None:
            # simply use the keys order (already set)
            self.slot_order = list(self.slot_states.keys())
        else:
            # preserve configured order, append any extra states that were added above
            ordered = [s for s in config_order if s in self.slot_states]
            for s in self.slot_states:
                if s not in ordered:
                    ordered.append(s)
            self.slot_order = ordered
        # --- END NEW

        self.suggest_state = states.get('suggest_state', 'suggest')
        self.no_alts_state = states.get('no_alts_state', 'no_alts')
        self.end_state = states.get('end_state', 'end')
        self.init_state = states.get('init_state', 'init')

        # dialog bookkeeping
        self.current_state = self.init_state
        # ensure preferences always has the expected keys (dynamically include any new slots)
        self.preferences = {v: None for v in self.slot_states.values()}
        self.text_columns = list(self.slot_states.values())
        self.suggest_counter = 0

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
    "welcome": "Welcome, what type of restaurant are you looking for!",
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
        for state in self.slot_order:
            slot = self.slot_states.get(state)
            if slot is None:
                continue
            if self.preferences.get(slot) is None:
                return state, self.templates.get(state, self.templates.get(f"ask_{slot}", ""))

        suggestions = self.sugg_engine.get_suggestions(self.preferences)
        if len(suggestions) > 0:
            suggestion_name = suggestions.iloc[self.suggest_counter % len(suggestions)]['restaurantname']
            self.suggest_counter += 1
            return self.suggest_state, f"How about {suggestion_name}?"
        return self.no_alts_state, self.templates.get('no_alts', "I couldn't find matches")

    def _handle_slot_state(self, state: str, utterance: str, dialog_act: str) -> Tuple[str, str]:
        slot = self.slot_states[state]
        if dialog_act == 'inform':
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

        if current_state == self.suggest_state:
            if dialog_act in ['bye', 'ack', 'affirm']:
                return self.end_state, self.templates.get('goodbye', 'Goodbye')
            suggestions = self.sugg_engine.get_suggestions(self.preferences)
            if len(suggestions) > 0 and self.suggest_counter < len(suggestions):
                suggestion_name = suggestions.iloc[self.suggest_counter]['restaurantname']
                self.suggest_counter += 1
                return self.suggest_state, f"How about {suggestion_name}?"
            return self.no_alts_state, self.templates.get('ask_change_slot', 'No alternatives')

        if current_state == self.no_alts_state:
            if dialog_act in ['affirm', 'ack']:
                self.preferences = {v: None for v in self.slot_states.values()}
                self.suggest_counter = 0
                return self.next_missing_state()

            slot, value = self.parser.parse_change_request(utterance)
            if slot == 'restart':
                self.preferences = {v: None for v in self.slot_states.values()}
                self.suggest_counter = 0
                return self.next_missing_state()[0], self.templates.get('reset_confirm', 'Restarting') + " " + self.templates.get(self.next_missing_state()[0], '')

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

    def run_dialog(self):
        print(self.templates.get('welcome', 'Welcome'))
        self.current_state = self.init_state
        while self.current_state != self.end_state:
            user_input = input('User: ')
            if user_input.lower() in ['quit', 'exit']:
                break
            next_state, system_response = self.state_transition(self.current_state, user_input)
            print(self.preferences)
            self.current_state = next_state
            print(f"System: {system_response}")
