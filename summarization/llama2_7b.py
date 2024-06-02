from .method import LlmMethod


class Llama2_7B(LlmMethod):
    def generate(self, txt: str) -> str:
        raise NotImplementedError()
