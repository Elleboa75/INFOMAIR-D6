import re
from typing import Tuple, Optional
from interfaces.parser_interface import ChangeRequestParserBase

class ChangeRequestParser(ChangeRequestParserBase):
    def _normalize_slot(self, raw: str) -> Optional[str]:
        raw = raw.lower().strip()
        if raw in ["area", "location", "neighborhood", "neighbourhood"]:
            return "area"
        if raw in ["food", "cuisine", "type"]:
            return "food"
        if raw in ["price", "pricerange", "budget", "cost", "pricing", "range", "price range"]:
            return "pricerange"
        # new mappings for added slots
        if raw in ["quality", "food quality", "food_quality", "rating"]:
            return "food_quality"
        if raw in ["crowdedness", "crowded", "busy", "not busy", "occupancy", "crowd"]:
            return "crowdedness"
        if raw in ["length_of_stay", "length of stay", "length", "stay", "duration"]:
            return "length_of_stay"
        return None

    def parse_change_request(self, utterance: str) -> Tuple[Optional[str], Optional[str]]:
        u = (utterance or '').lower().strip()
        if any(k in u for k in ["restart", "start over", "reset", "change all", "everything"]):
            return ("restart", None)

        # If user indicates a slot to change without providing a value
        # include synonyms for the new slots in the key list
        for key in [
            "area", "food", "price", "pricerange", "budget", "price range",
            "quality", "food quality", "crowdedness", "crowded", "busy",
            "length", "length of stay", "stay", "duration"
        ]:
            if key in u and (" to " not in u) and ("=" not in u):
                return (self._normalize_slot(key), None)

        # regex to capture "slot to value" or "slot = value"
        # allow multi-word values (e.g., "not busy", "very good")
        pattern = re.compile(
            r"(area|food|price|pricerange|price range|quality|food quality|food_quality|crowdedness|crowded|busy|not busy|length|length of stay|length_of_stay|stay|duration)\s*(?:to|=)\s*([a-zA-Z\-\s]+)",
            re.IGNORECASE
        )
        m = pattern.search(u)
        if m:
            slot = self._normalize_slot(m.group(1))
            value = m.group(2).strip()
            # normalize whitespace in value
            value = re.sub(r"\s+", " ", value)
            return (slot, value)

        # global 'any' preference
        if u in ["any", "whatever", "no preference"]:
            return (None, "any")

        return (None, None)
