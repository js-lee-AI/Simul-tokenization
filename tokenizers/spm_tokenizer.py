from base_tokenizer import MetaTokenizer

from typing import List

import sentencepiece as spm


class SPMTokenizer(MetaTokenizer):
    """
    Sentencepiece tokenizer.
    Unigram, BPE, Word and Character (de)tokenization
    """

    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def tokenize(self, input_text: str) -> List[str]:
        tokens = self.sp.encode_as_pieces(input_text.strip())
        return tokens

    def detokenize(self, output_tokens: List[str]) -> str:
        detokenized_text = "".join(output_tokens).replace("▁", " ").strip()
        return detokenized_text


if __name__ == "__main__":
    path_ko_en_model = "./bpe/shared_bpe_8k.model"

    ko_text = "텍스트는 대표적인 비정형 데이터 이므로 텍스트 그 자체로 분석을 할 수 없어 일정부분 정형화된 형태를 사용합니다."
    en_text = "Since most texts represent unstructured data, it cannot be analyzed by itself, so a standardized form is used to some extent."
    spm_tokenizer = SPMTokenizer(path_ko_en_model)
    ko_tokens = spm_tokenizer.tokenize(ko_text)
    en_tokens = spm_tokenizer.tokenize(en_text)
    ko_detokenized_tokens = spm_tokenizer.detokenize(ko_tokens)
    en_detokenized_tokens = spm_tokenizer.detokenize(en_tokens)

    print(
        f"original koran text: {ko_text}\ntokenized tokens: {ko_tokens}\ndetokenized text: {ko_detokenized_tokens}\n"
    )
    print(
        f"original english text: {en_text}\ntokenized tokens: {en_tokens}\n, detokenized text: {en_detokenized_tokens}\n"
    )

    print(en_tokens)
