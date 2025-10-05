from abc import ABC, abstractmethod
from typing import Tuple, Optional

class ChangeRequestParserBase(ABC):
    @abstractmethod
    def parse_change_request(self, utterance: str) -> Tuple[Optional[str], Optional[str]]:
        raise NotImplementedError
