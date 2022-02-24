import os
from typing import List

from konlpy.tag import Mecab
from tqdm import tqdm

from simultokenizer.utils.train_spm_tokenizer import SPMTrainer
from simultokenizer.utils.module import TokenizerModule


class MorphemeTrainer:
    def __init__(
        self,
        corpus_path="./dataset/en-ko",
        morphs_output_corpus_name="aihub_ko_morpheme.txt",
        cfg_path=None,
        cfg_name=None,
    ):
        self.corpus_path = corpus_path
        self.morphs_output_corpus_name = morphs_output_corpus_name
        self.morphs_output_corpus_path = os.path.join(corpus_path, morphs_output_corpus_name)
        self.output_corpus = None

        # super(MorphemeTrainer, self).__init__()
        self.cfg_path = cfg_path
        self.cfg_name = cfg_name
        # self.compose_configure()

    @staticmethod
    def replace_whitespace(texts: List[str]) -> List[str]:
        """
        Replace whitespace with '★'
        """
        modified_text = []
        print("Converting whitespace in your corpus...")
        for text in tqdm(texts):
            modified_text.append(text.rstrip().replace(" ", "★"))
        return modified_text

    @staticmethod
    def segment_morpheme(texts: List[str]) -> List[str]:
        print("Converting to morpheme-based corpus...")
        m = Mecab()
        morpheme_aware_texts = []
        for text in tqdm(texts):
            morpheme_aware_texts.append(" ".join(m.morphs(text.rstrip())) + "\n")
        return morpheme_aware_texts

    def segment_corpus(self):
        with open(self.corpus_path, "r", encoding="utf8") as f:
            texts = f.readlines()

        texts = self.replace_whitespace(texts)
        texts = self.segment_morpheme(texts)

        # print(texts[:3])
        self.output_corpus = "".join(texts)
        with open(self.morphs_output_corpus_path, "w", encoding="utf8") as f:
            f.write(self.output_corpus)

    def train(self):
        # Build a new morpheme-based corpus
        if not os.path.isfile(self.morphs_output_corpus_path):
            self.segment_corpus()
        # Use existing morpheme-based corpus
        else:
            with open(self.morphs_output_corpus_path, "r", encoding="utf8") as f:
                self.output_corpus = f.readlines()

        trainer = SPMTrainer(
            cfg_path=self.cfg_path,
            cfg_name=self.cfg_name,
            output_morphs_corpus_path=self.morphs_output_corpus_name,
        )
        trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hydra configuration file")
    parser.add_argument(
        "--cfg-name",
        "-c",
        type=str,
        default="morpheme_aware_subword.yaml",
        help="hydra config file name (.yalm)",
    )
    args = parser.parse_args()

    trainer = MorphemeTrainer(
        cfg_path="../config",
        cfg_name=args.cfg_name,
    )
    trainer.train()
