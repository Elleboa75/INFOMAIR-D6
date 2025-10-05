import re
import json
import os
from typing import Dict, Any, List, Tuple
from interfaces.extractor_interface import SlotExtractorBase

class SlotExtractor(SlotExtractorBase):
    """Config-driven slot extractor.

    Loads patterns from a JSON-like config (same shape used by the DialogManager)
    and exposes `extract_preferences(utterance)`.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.compiled_patterns: Dict[str, List[Tuple[re.Pattern, int]]] = {}
        self._compile_patterns()

    @classmethod
    def from_config_file(cls, path: str):
        if path and os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        else:
            cfg = {}
        return cls(cfg)

    def _compile_patterns(self):
        raw_patterns = self.config.get('patterns', {})
        # fallback to a safe default if not provided
        if not raw_patterns:
            raw_patterns = {
                'food': [], 'area': [], 'pricerange': [], 'food_quality': [], 'crowdedness': [], 'length_of_stay' : []
            }

        for slot, slot_list in raw_patterns.items():
            self.compiled_patterns[slot] = []
            for entry in slot_list:
                if isinstance(entry, str):
                    pattern = entry
                    group = 1
                else:
                    pattern = entry.get('pattern')
                    group = entry.get('group', 1)
                try:
                    self.compiled_patterns[slot].append((re.compile(pattern, re.I), group))
                except re.error:
                    # ignore invalid regex
                    continue

    def extract_preferences(self, utterance: str) -> Dict[str, str]:
        u = (utterance or '').lower()
        extracted: Dict[str, str] = {}
        for slot, patterns in self.compiled_patterns.items():
            for pattern, group in patterns:
                m = pattern.search(u)
                if m:
                    try:
                        val = m.group(group)
                        if val:
                            extracted[slot] = val.lower()
                            break
                    except IndexError:
                        continue
        return extracted
