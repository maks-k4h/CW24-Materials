from enum import Enum

from . import gemma_2b, llama2_7b, llama3_8b


class Method(Enum):
    Llama3_8B = 1
    Llama2_7B = 2
    Gemma_2B = 3


class LlmSummarizationContext:
    def __init__(
            self,
            method: Method,
            hf_token: str | None = None
    ) -> None:
        self.method = method
        self._llm = {
            Method.Llama3_8B: llama3_8b.Llama3_8B,
            Method.Llama2_7B: llama2_7b.Llama2_7B,
            Method.Gemma_2B: gemma_2b.Gemma2B,
        }[self.method](hf_token=hf_token)

    def summarize(self, text: str) -> str:
        instruct = "Summarize the following lyrics:\n" + text
        return self._llm.generate(instruct)



