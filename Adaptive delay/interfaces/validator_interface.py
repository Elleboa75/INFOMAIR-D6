from abc import ABC, abstractmethod
from typing import Dict, Any

class PreferenceValidatorBase(ABC):
    @abstractmethod
    def register_validator(self, slot: str, func):
        raise NotImplementedError

    @abstractmethod
    def check_preference_validity(self, preferences: Dict[str, str]):
        raise NotImplementedError
