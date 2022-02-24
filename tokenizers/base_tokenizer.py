from typing import List

from abc import *


class MetaTokenizer(metaclass=ABCMeta):
    """
    Tokenizer Meta Class
    """

    @abstractmethod
    def tokenize(self, input_text: str) -> List[str]:
        pass

    @abstractmethod
    def detokenize(self, output_tokens: List[str]) -> str:
        pass
