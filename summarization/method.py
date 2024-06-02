from abc import ABC, abstractmethod

import torch.cuda


class LlmMethod(ABC):
    def __init__(self):
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self._device = torch.device('mps')
        else:
            self._device = torch.device('cpu')

    @abstractmethod
    def generate(self, txt: str) -> str:
        pass
