from abc import ABC, abstractmethod
from typing import Dict

class SlotExtractorBase(ABC):
    @abstractmethod
    def extract_preferences(self, utterance: str) -> Dict[str, str]:
        """Extract slot->value map from the user utterance.
        """
        raise NotImplementedError