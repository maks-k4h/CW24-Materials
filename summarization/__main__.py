from argparse import ArgumentParser

from .llm import Method, LlmSummarizationContext


str2method = {
    'gemma_2b': Method.Gemma_2B,
    'llama2_7b': Method.Llama2_7B,
    'llama3_8b': Method.Llama3_8B,
}


def main(
        lyrics: str,
        method: str,
        hf_token: str | None = None,
) -> None:
    assert method in str2method
    ctx = LlmSummarizationContext(method=str2method[method])
    output = ctx.summarize(lyrics)
    print(output)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--lyrics', required=True)
    parser.add_argument('--model', choices=str2method.keys(), required=True)
    parser.add_argument('--hf-token', required=False, default=None,
                        help='HuggingFace token (or use env. var. HF_TOKEN)')

    args = parser.parse_args()

    main(
        lyrics=args.lyrics,
        method=args.model,
        hf_token=args.hf_token,
    )

