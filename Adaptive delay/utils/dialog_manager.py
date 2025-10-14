
from typing import Tuple, Dict, Any
import json
import os
import time
import re
import numpy as np
import pandas as pd

# ---- Robust imports: try relative, then absolute ----
try:
    # If this file is inside the utils/ package
    from .restaurant_matcher import RestaurantMatcher
    from .extractor import SlotExtractor
    from .validator import PreferenceValidator
    from .suggestion_engine import SuggestionEngine
    from .parser import ChangeRequestParser
    from interfaces.dialog_manager_interface import DialogManagerBase  # adjust if this is elsewhere
    import models.baseline as baseline
except Exception:
    # If importing relatively fails (running as a script), try absolute
    from utils.restaurant_matcher import RestaurantMatcher
    from utils.extractor import SlotExtractor
    from utils.validator import PreferenceValidator
    from utils.suggestion_engine import SuggestionEngine
    from utils.parser import ChangeRequestParser
    from interfaces.dialog_manager_interface import DialogManagerBase  # adjust if needed
    import models.baseline as baseline


class DialogManager(DialogManagerBase):
    """Coordinator that composes the small components (extractor, validator,
    suggestion engine, parser) and implements the dialog policy.

    Extended with:
      - Adaptive delay (based on reply length, compute latency, urgency)
      - Typing indicator during the delay
      - Word-by-word streaming of the response
    """

    def __init__(self,
                 df: pd.DataFrame,
                 config_path: str = "dialog_config.json",
                 model: Any = None,
                 extractor: SlotExtractor = None,
                 validator: PreferenceValidator = None,
                 sugg_engine: SuggestionEngine = None,
                 parser: ChangeRequestParser = None,
                 all_caps: bool = False,
                 allow_restarts: bool = True,
                 delay: int = 0,
                 formal: bool = True,
                 # NEW:
                 adaptive_delay: bool = True,
                 user_wps: float = 3.0):

        # Initialize df and config variables
        self.df = df
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.all_caps = all_caps
        self.allow_restart = allow_restarts
        self.delay = delay  # used if adaptive_delay=False
        self.adaptive_delay = adaptive_delay  # toggle
        self.user_wps = user_wps  # words per second for streaming
        self.last_suggested = None

        # Load the response templates and state configs
        if formal:
            self.templates = self.config.get('formal_templates', {})
        else:
            self.templates = self.config.get("informal_templates", {})

        states = self.config.get('states', {})
        self.slot_states = states.get('slot_states', {})

        # Preference slots to extract from the user utterances
        extra_slots = [
            ('ask_food_quality', 'foodquality'),
            ('ask_crowdedness', 'crowdedness'),
            ('ask_length_of_stay', 'lengthstay'),
        ]
        for state_name, slot_name in extra_slots:
            if slot_name in self.df.columns and slot_name not in self.slot_states.values():
                self.slot_states[state_name] = slot_name

        # Basic slots (must be filled), extra slots are optional
        self.basic_slots = ['ask_area', 'ask_food', 'ask_pricerange']
        self.extra_slots = [s for s in self.slot_states.keys() if s not in self.basic_slots]

        # slot order: basic slots first, then extra slots
        self.slot_order = [s for s in self.basic_slots if s in self.slot_states] + self.extra_slots

        self.suggest_state = states.get('suggest_state', 'suggest')
        self.no_alts_state = states.get('no_alts_state', 'no_alts')
        self.additional_req_state = states.get('additional_req_state', 'add_req')
        self.end_state = states.get('end_state', 'end')
        self.init_state = states.get('init_state', 'init')
        self.req_state = states.get('req_state', 'req_state')

        # dialog bookkeeping
        self.current_state = self.init_state

        # ensure preferences always has the expected keys (dynamically include any new slots)
        self.preferences = {v: None for v in self.slot_states.values()}
        self.extra_preferences = None
        self.text_columns = list(self.slot_states.values())
        self.suggest_counter = 0

        # components (allow dependency injection for testing)
        self.extractor = extractor or SlotExtractor.from_config_file(config_path)
        self.validator = validator or PreferenceValidator(self.df)
        self.sugg_engine = sugg_engine or SuggestionEngine(self.df)
        self.parser = parser or ChangeRequestParser()

        # model loading for dialog act classification
        self.model = model
        if self.model is None:
            try:
                self.model = baseline.Train_Baseline_2()
            except Exception:
                self.model = None

    # -------------------------
    # Config loader
    # -------------------------
    def _load_config(self, path: str) -> Dict[str, Any]:
        """
        Load the config.json file if it exists, otherwise return a standard config file
        """
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

    # -------------------------
    # Adaptive delay helpers
    # -------------------------
    def _estimate_tokens(self, text: str) -> int:
        words = len((text or "").strip().split())
        return max(1, round(words * 0.75))

    def _detect_urgency(self, user_text: str) -> bool:
        u = (user_text or "").strip().lower()
        if not u:
            return False
        if len(u) <= 2:  # "?", "ok"
            return True
        if "??" in u or re.search(r"\b(now|quick|urgent|hurry|asap)\b", u):
            return True
        return False

    def adaptive_delay_ms(self, reply_text: str, model_latency_ms: float = 0.0, urgency: bool = False) -> int:
        base = 150  # ms
        min_d = 120
        max_d = 2500
        k_len = 15  # ms per token
        k_lat = 0.2  # fraction of compute latency reflected
        k_read = 5  # ms per token for pacing
        k_urg = 800  # urgency discount (ms)

        tokens = self._estimate_tokens(reply_text)
        delay = base + k_len * tokens + k_lat * model_latency_ms + k_read * tokens
        if urgency:
            delay -= k_urg

        if tokens < 10:
            delay = min(max(delay, min_d), 700)

        return int(max(min_d, min(delay, max_d)))

    # -------------------------
    # Typing indicator + streaming
    # -------------------------
    def show_thinking(self, duration_s: float = None):
        """Typing indicator animation that lasts `duration_s` seconds."""
        if duration_s is None:
            duration_s = float(self.delay)

        end_time = time.perf_counter() + max(0.0, duration_s)
        frames = ["Typing   ", "Typing.  ", "Typing.. ", "Typing..."]
        i = 0
        while time.perf_counter() < end_time:
            print("\r" + frames[i % len(frames)], end="", flush=True)
            time.sleep(0.2)
            i += 1
        print("\r" + " " * 20 + "\r", end="", flush=True)

    def stream_response(self, text: str):
        """Stream the system response letter-by-letter."""
        if not text:
            print()
            return

        # pacing knobs
        base_char_delay = 1.0 / (self.user_wps * 5)  # ~5 chars per word
        min_delay = 0.005
        max_delay = 0.05

        for ch in text:
            out = ch.upper() if self.all_caps else ch
            # write at the lowest level to avoid buffering issues
            import sys
            sys.stdout.write(out)
            sys.stdout.flush()

            # small natural variation
            if ch in ".,;!?":
                pause = base_char_delay * 4
            elif ch == " ":
                pause = base_char_delay * 2
            else:
                pause = base_char_delay

            time.sleep(max(min_delay, min(pause, max_delay)))

        print()  # newline at the end

    # -------------------------
    # NLU + state machine
    # -------------------------
    def classify_dialog_act(self, utterance: str) -> str:
        """
        Classify the user utterance using the classifcation model provided, if no model is provided use a very simple model instead
        """
        u = (utterance or '').lower().strip()
        if self.model is None:
            if any(w in u for w in ["hello", "hi", "hey"]):
                return "hello"
            if any(w in u for w in ["bye", "goodbye"]):
                return "bye"
            return "inform"
        try:
            res = self.model.predict([u])
            if isinstance(res, dict) and 'predicted_dialog_act' in res:
                return res['predicted_dialog_act']
            if hasattr(self.model, 'predict_label'):
                return self.model.predict_label(u)
            if isinstance(res, str):
                return res
        except Exception:
            pass
        return "null"

    def next_missing_state(self) -> Tuple[str, str]:
        """
        Looks for the next missing preference, if all preferences are filled, move to additional request or suggest state
        """
        # Check if basic slots (area, food, pricerange) are all filled
        basic_slots_filled = all(
            self.preferences.get(self.slot_states.get(state)) is not None
            for state in self.basic_slots
            if state in self.slot_states
        )

        # If basic slots are filled and we have extra slots available and haven't asked yet
        if basic_slots_filled and self.extra_slots and self.extra_preferences == None:
            return self.additional_req_state, self.templates.get('additional_req',
                                                                 "Do you have any additional requirements?")

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
        """
        Function that handles the basic preference slot states (area / food / price range)
        """
        slot = self.slot_states[state]
        if dialog_act in ['inform', 'null']:  ## added because the dialog act makes mistakes
            if 'any' in (utterance.lower() or '').split():
                self.preferences[slot] = 'any'
                return self.next_missing_state()

            extracted = self.extractor.extract_preferences(utterance)
            # slot-specific single-token fallback (when system is explicitly asking a slot)
            if not extracted or slot not in extracted:
                try:
                    lowest_distance_index = []
                    matches = []
                    for word in utterance.split():
                        match = RestaurantMatcher(self.df, self.text_columns, word).match()
                        best = getattr(match, 'best_match', None)
                        if (self.df[slot].str.lower() == best).any():
                            matches.append(best)
                            lowest_distance_index.append(match.distance)
                        else:
                            matches.append('None')
                            lowest_distance_index.append(10e5)
                    if lowest_distance_index[np.argmin(lowest_distance_index)] < 10e5:
                        self.preferences[slot] = matches[np.argmin(lowest_distance_index)]
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
        """
        Main dialog state flow handler, decides what happens in each state
        """
        utterance = (user_utterance or '').lower()
        dialog_act = self.classify_dialog_act(utterance)
        if current_state == self.init_state:
            # Stay in the same state if the user greets the system
            if dialog_act == 'hello':
                return self.init_state, self.templates.get('welcome', 'Welcome')

            # If the user wants to inform the system, extract the preferences
            if dialog_act == 'inform':
                # Extract preferences using RegEx
                extracted = self.extractor.extract_preferences(utterance)

                if not extracted:
                    tokens = utterance.strip().split()
                    if len(tokens) == 1 and tokens[0]:
                        tok = tokens[0].lower()
                        priority_cols = [c for c in
                                         ['food', 'area', 'pricerange', 'food_quality', 'crowdedness', 'length_of_stay']
                                         if c in self.text_columns]
                        for col in priority_cols:
                            exact_matches = self.df[col].str.lower() == tok
                            if exact_matches.any():
                                extracted[col] = tok

                # Check the validity of the extracted preferences (are they in to dataframe)
                validity = self.validator.check_preference_validity(extracted)
                if validity is True:
                    self.preferences.update(extracted)
                    return self.next_missing_state()

                return self.next_missing_state()
            return self.init_state, self.templates.get('welcome', 'Welcome')

        if current_state in self.slot_states:
            # Uses the _handle_slot_state function for the basic preferences
            return self._handle_slot_state(current_state, user_utterance, dialog_act)

        if current_state == self.additional_req_state:
            # In this state we are handling extra request (romantic / touristic / children)
            if dialog_act in ['affirm', 'ack']:
                # If the user acknowledges extra requests are present, prompt the user to state said requests
                return self.additional_req_state, self.templates.get('request', 'Please state your requests')

            # If the user informs the system about extra requests, extract them and create suggestions based on the requests
            elif dialog_act in ['inform']:
                request_strings = ['touristic', 'assigned seats', 'children', 'romantic']
                actual_requests = []
                for req_string in request_strings:
                    if req_string in user_utterance.lower():
                        actual_requests.append(req_string)
                if len(actual_requests) == 0:
                    return self.additional_req_state, self.templates.get('additional_req',
                                                                         "Do you have any additional requirements?")
                self.extra_preferences = actual_requests
                suggestions = self.sugg_engine.get_suggestions_extra_req(self.preferences, actual_requests)
                if len(suggestions) > 0:
                    suggestion_name = suggestions.iloc[self.suggest_counter % len(suggestions)]['restaurantname']
                    self.last_suggested = suggestion_name
                    self.suggest_counter += 1
                    return self.suggest_state, f"How about {suggestion_name}?"

                return self.no_alts_state, self.templates.get('no_alts', "I couldn't find matches")

            # If the user has no extra requests make suggestions based on current preferences
            elif dialog_act in ['negate', 'deny']:
                suggestions = self.sugg_engine.get_suggestions(self.preferences)
                if len(suggestions) > 0:
                    suggestion_name = suggestions.iloc[self.suggest_counter % len(suggestions)]['restaurantname']
                    self.last_suggested = suggestion_name
                    self.suggest_counter += 1
                    return self.suggest_state, f"How about {suggestion_name}?"
                return self.no_alts_state, self.templates.get('no_alts', "I couldn't find matches")

            # If the dialog_act can't be classified into the three forms above
            else:
                return self.additional_req_state, self.templates.get('additional_req',
                                                                     "Do you have any additional requirements?")

        if current_state == self.suggest_state:
            # Once all preferences + extra requirements are extracted, make suggestions
            # Provide contact information upon request of user
            if dialog_act == "request":
                if self.last_suggested:
                    return self.provide_contact_info(self.last_suggested)

            # Terminate the dialog once the user says goodbye
            if dialog_act in ['bye', 'ack', 'affirm']:
                return self.end_state, self.templates.get('goodbye', 'Goodbye')

            # Make suggestions based on preferences + extra requirements
            if self.extra_preferences != None:
                suggestions = self.sugg_engine.get_suggestions_extra_req(self.preferences, self.extra_preferences)

            # Make suggestions based on preferences alone
            else:
                suggestions = self.sugg_engine.get_suggestions(self.preferences)

            # Choose a suggestion that is not chosen yet
            if len(suggestions) > 0 and self.suggest_counter < len(suggestions):
                suggestion_name = suggestions.iloc[self.suggest_counter]['restaurantname']
                self.last_suggested = suggestion_name
                self.suggest_counter += 1
                return self.suggest_state, f"How about {suggestion_name}?"

            # If not suggestions are available, ask the user for a change in preference
            return self.no_alts_state, self.templates.get('ask_change_slot', 'No alternatives')

        if current_state == self.no_alts_state:
            # If the user wants to reset the preferences, reset them and move to the first required slot state and extract the others sequentially
            if dialog_act in ['affirm', 'ack']:
                self.preferences = {v: None for v in self.slot_states.values()}
                self.suggest_counter = 0
                self.extra_preferences = None
                return self.next_missing_state()

            # Parse the user utterance for specific preference changes
            slot, value = self.parser.parse_change_request(utterance)

            # If the user ask for a restart, completly restart the system (move to the init state) <-- This is different from the reset above, in this case you can state 3 preferences at the same time
            if slot == 'restart':
                if self.allow_restart == True:
                    self.preferences = {v: None for v in self.slot_states.values()}
                    self.suggest_counter = 0
                    self.extra_preferences = None

                    return self.init_state, self.templates.get('reset_confirm',
                                                               'Restarting') + " " + self.templates.get(
                        self.next_missing_state()[0], '')
                else:
                    return self.no_alts_state, self.templates.get('no_reset', 'No resets allowed')

            if slot is None and value == 'any':
                for s in self.text_columns:
                    if self.preferences.get(s) not in [None, 'any']:
                        self.preferences[s] = 'any'
                        self.suggest_counter = 0
                        return self.next_missing_state()
                return self.no_alts_state, self.templates.get('ask_change_slot', 'Which slot to change?')

            # If the user states a specific preference to change, use the slot, reset it and move to the corrosponding state.
            if slot in self.text_columns:
                if value is None:
                    self.preferences[slot] = None
                    self.suggest_counter = 0
                    prompt_slot = 'price range' if slot == 'pricerange' else slot
                    state_name = next((s for s, sl in self.slot_states.items() if sl == slot), None)
                    return state_name, self.templates.get('confirm_change', "Okay—let's update {slot}.").format(
                        slot=prompt_slot)
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
            # Termination of dialog state
            return self.end_state, self.templates.get('goodbye', 'Goodbye')

        # Basic fallback when something is not correctly identified or extracted
        return current_state, 'Could you please rephrase?'

    def provide_contact_info(self, restaurant_name):
        # Function to provide contact information for the last suggested restaurant
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

    # -------------------------
    # UI/UX: typing + adaptive delay + streaming
    # -------------------------
    def run_dialog(self):
        # Function to start the dialog loop
        welcome = self.templates.get('welcome', 'Welcome')
        print(welcome.upper() if self.all_caps else welcome)

        self.current_state = self.init_state
        while self.current_state != self.end_state:
            user_input = input('User: ')
            if user_input.lower() in ['quit', 'exit']:
                break

            # measure "compute" latency for realism
            t0 = time.perf_counter()
            next_state, system_response = self.state_transition(self.current_state, user_input)
            compute_ms = (time.perf_counter() - t0) * 1000.0
            self.current_state = next_state

            # decide delay
            if self.adaptive_delay:
                urgency = self._detect_urgency(user_input)
                delay_ms = self.adaptive_delay_ms(system_response, compute_ms, urgency)
                delay_s = delay_ms / 1000.0
            else:
                delay_s = float(self.delay)

            # typing indicator
            self.show_thinking(delay_s)

            # stream response word by word
            prefix = "System: "
            print(prefix.upper() if self.all_caps else prefix, end="", flush=True)
            self.stream_response(system_response)
