from .method import LlmMethod


class Llama3_8B(LlmMethod):
    def generate(self, txt: str) -> str:
        raise NotImplementedError()
