from .method import LlmMethod

import torch
import transformers

MODEL_CHECKPOINT = 'google/gemma-2b'


class Gemma2B(LlmMethod):
    def __init__(self, hf_token: str):
        super().__init__()
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, hf_token=hf_token)
        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_CHECKPOINT,
            torch_dtype=torch.float16,
            token=hf_token,
        )

        self._generation_config = transformers.GenerationConfig(max_length=8_000)

    def generate(self, txt: str) -> str:
        input_text = txt
        input_ids = self._tokenizer(input_text, return_tensors='pt')
        outputs = self._model.generate(**input_ids, generation_config=self._generation_config)
        output_text = self._tokenizer.decode(outputs[0])
        return output_text
