from abc import ABC, abstractmethod
from typing import Tuple

class DialogManagerBase(ABC):
    """Abstract interface for a DialogManager implementation."""

    @abstractmethod
    def state_transition(self, current_state: str, user_utterance: str) -> Tuple[str, str]:
        """Process the user's utterance given current_state and return (next_state, response)."""
        raise NotImplementedError

    @abstractmethod
    def run_dialog(self) -> None:
        """Run an interactive dialog loop"""
        raise NotImplementedError

    def next_missing_state(self) -> Tuple[str, str]:
        """Optional: return the next missing state and prompt."""
        raise NotImplementedError

    def register_validator(self, slot: str, func) -> None:
        """Optional: register a custom validator for a slot."""
        raise NotImplementedError
