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
        return None

    def parse_change_request(self, utterance: str) -> Tuple[Optional[str], Optional[str]]:
        u = (utterance or '').lower().strip()
        if any(k in u for k in ["restart", "start over", "reset", "change all", "everything"]):
            return ("restart", None)

        for key in ["area", "food", "price", "pricerange", "budget", "price range"]:
            if key in u and (" to " not in u) and ("=" not in u):
                return (self._normalize_slot(key), None)

        m = re.search(r"(area|food|price|pricerange|price range)\s*(?:to|=)\s*([a-zA-Z\-]+)", u)
        if m:
            slot = self._normalize_slot(m.group(1))
            value = m.group(2)
            return (slot, value)

        if u in ["any", "whatever", "no preference"]:
            return (None, "any")

        return (None, None)
