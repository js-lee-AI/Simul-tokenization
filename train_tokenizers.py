import os
import argparse
import time
import hydra

import sentencepiece as spm
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig


class SPMTrainer:
    def __init__(
        self,
        cfg_path="./config",
        cfg_name="vocab_baseline.yaml",
    ):
        self.cfg_path_and_name = os.path.join(cfg_path, cfg_name)
        hydra.initialize(cfg_path, cfg_name)

        self.config = hydra.compose(cfg_name)
        self.multiple_vocabsize = False

        # encoder/decoder uses same tokenization methods
        if isinstance(self.config.path.corpus_name, str):
            self.path_corpus = os.path.join(
                self.config.path.data_dir, self.config.path.corpus_name
            )
            self.split_srctgt = False
        # encoder/decoder uses different tokenization methods
        elif isinstance(self.config.path.corpus_name, DictConfig):
            self.path_corpus = {
                "src": os.path.join(
                    self.config.path.data_dir, self.config.path.corpus_name["src"]
                ),
                "tgt": os.path.join(
                    self.config.path.data_dir, self.config.path.corpus_name["tgt"]
                ),
            }
            self.split_srctgt = True
        else:
            raise ValueError(
                f"'{self.config.path.corpus_name}' must be either string or dictionary, \not {type(self.config.path.corpus_name)}"
            )

        self.print_args()
        self.check_config()
        self.move_to_output_path()

        self.prefix = self.set_tokenizer_prefix(
            multiple_flag=self.multiple_vocabsize,
            vocab_sizes=self.config.tokenizer.vocab_size,
            cfg_name=cfg_name,
            corpus_name=self.config.path.corpus_name,
            vocab_type=self.config.tokenizer.vocab_type,
            vocab_languages=self.config.tokenizer.vocab_languages,
        )

    def print_args(self):
        print(f"Your config file: {self.cfg_path_and_name}")
        print(f"Your config: ")
        time.sleep(7)

    def check_config(self):
        available_vocab_types = ["unigram", "bpe", "word", "char"]

        if isinstance(self.config.tokenizer.vocab_size, ListConfig):
            self.multiple_vocabsize = True

        if isinstance(self.config.tokenizer.vocab_type, str):
            if self.config.tokenizer.vocab_type not in available_vocab_types:
                raise ValueError(
                    f"'{self.config.tokenizer.vocab_type}' is not in {available_vocab_types}."
                )
        elif isinstance(self.config.tokenizer.vocab_type, DictConfig):
            for vocab_type in self.config.tokenizer.vocab_type.values():
                if vocab_type not in available_vocab_types:
                    raise ValueError(
                        f"'{self.config.tokenizer.vocab_type}' is not in {available_vocab_types}."
                    )

        if not os.path.isdir(self.config.path.data_dir):
            raise OSError(
                "'{}' directory is not exist.".format(self.config.path.data_dir)
            )

        if self.split_srctgt is True:
            for idx, (key, path) in enumerate(self.path_corpus.items()):
                if not os.path.isfile(path):
                    raise OSError("'{}' file is not exist.".format(path))
                else:
                    print(f'vocab training courpus Files; {idx+1}) {key}: "{path}"')
                # get abstract paths
                self.path_corpus[key] = os.path.abspath(self.path_corpus[key])
        else:
            if not os.path.isfile(self.path_corpus):
                raise FileNotFoundError(
                    '"{}" file is not exist.'.format(self.path_corpus)
                )
            else:
                print(f'vocab training courpus file: "{self.path_corpus}"')
            # get abstract paths
            self.path_corpus = os.path.abspath(self.path_corpus)

    def move_to_output_path(self):
        os.makedirs(self.config.path.output_path, exist_ok=True)
        os.chdir(os.path.abspath(self.config.path.output_path))

    @staticmethod
    def set_tokenizer_prefix(
        multiple_flag,
        vocab_sizes,
        cfg_name,
        corpus_name,
        vocab_type,
        vocab_languages,
    ):
        # encoder/decoder use not same tokenization method
        if isinstance(corpus_name, DictConfig):
            prefix = dict()
            prefix["src"] = [
                vocab_languages["src"]
                + "_"
                + vocab_type["src"]
                + "_"
                + str(vocab_size[0] // 1000)
                + "k"
                for vocab_size in vocab_sizes
            ]
            prefix["tgt"] = [
                vocab_languages["tgt"]
                + "_"
                + vocab_type["tgt"]
                + "_"
                + str(vocab_size[1] // 1000)
                + "k"
                for vocab_size in vocab_sizes
            ]
            return prefix

        # encoder/decoder use same tokenization method
        elif isinstance(corpus_name, str):
            # multiple vocab size
            if multiple_flag is True:
                return [
                    vocab_languages
                    + "_"
                    + cfg_name[:-5]
                    + "_"
                    + str(vocab_size // 1000)
                    + "k"
                    for vocab_size in vocab_sizes
                ]
            # single vocab size
            else:
                return (
                    vocab_languages
                    + "_"
                    + cfg_name[:-5]
                    + "_"
                    + str(vocab_sizes // 1000)
                    + "k"
                )
        else:
            raise NotImplementedError

    def train(self):
        print("\n")
        print("Train SPM...")
        # training split tokenizers
        if self.split_srctgt is True:
            for idx, (key, prefixes) in enumerate(self.prefix.items()):
                keys = ["src", "tgt"]
                for prefix, vocab_size in zip(
                    prefixes, self.config.tokenizer.vocab_size
                ):
                    spm.SentencePieceTrainer.Train(
                        f"--input={self.path_corpus[keys[idx]]} \
                                                     --model_prefix={prefix} \
                                                     --vocab_size={vocab_size[idx]} \
                                                     --model_type={self.config.tokenizer.vocab_type[keys[idx]]} \
                                                     --num_threads={self.config.tokenizer.args.num_threads} \
                                                     --max_sentence_length={self.config.tokenizer.max_sentence_length}"
                    )
                    print(f"Training {prefix} tokenizer succeed")
        else:
            # training multiple tokenizers
            if self.multiple_vocabsize is True:
                for idx, prefix in enumerate(self.prefix):
                    spm.SentencePieceTrainer.Train(
                        f"--input={self.path_corpus} \
                                                     --model_prefix={prefix} \
                                                     --vocab_size={self.config.tokenizer.vocab_size[idx]} \
                                                     --model_type={self.config.tokenizer.vocab_type} \
                                                     --num_threads={self.config.tokenizer.args.num_threads} \
                                                     --max_sentence_length={self.config.tokenizer.max_sentence_length}"
                    )
                    print(f"Training {self.prefix} tokenizer succeed")
            # training single tokenizer
            else:
                spm.SentencePieceTrainer.Train(
                    f"--input={self.path_corpus} \
                                                 --model_prefix={self.prefix} \
                                                 --vocab_size={self.config.tokenizer.vocab_size} \
                                                 --model_type={self.config.tokenizer.vocab_type} \
                                                 --num_threads={self.config.tokenizer.args.num_threads} \
                                                 --max_sentence_length={self.config.tokenizer.max_sentence_length}"
                )
                print("Training tokenizer completely succeed")
        print("Finish...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hydra configuration file")
    parser.add_argument(
        "--cfg-name",
        "-c",
        type=str,
        default="vocab_baseline.yaml",
        help="hydra config file name (.yalm)",
    )
    args = parser.parse_args()

    trainer = SPMTrainer(
        cfg_path="./config",
        cfg_name=args.cfg_name,
    )
    trainer.train()
