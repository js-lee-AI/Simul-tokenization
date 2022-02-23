from typing import Union, List, Dict, Tuple, Any
import os
import argparse
import time
import hydra

import sentencepiece as spm
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig

import numpy as np

# Todo: To work regardless of the order of the 'src', 'tgt' keys in 'split_unigram_bpe.yaml' file
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
        multiple_flag: bool,
        vocab_sizes: Union[List[int], List[int], int],
        cfg_name: str,
        corpus_name: Union[Dict[str, str], str],
        vocab_type: Union[Dict[str, str], str],
        vocab_languages: Union[Dict[str, str], str],
    ) -> Union[List[str], str]:
        # encoder/decoder use not same tokenization method
        if isinstance(corpus_name, DictConfig):
            prefix = list()
            # source
            prefix.extend(
                [
                    vocab_languages["src"]
                    + "_"
                    + vocab_type["src"]
                    + "_"
                    + str(vocab_size[0] // 1000)
                    + "k"
                    for vocab_size in vocab_sizes
                ]
            )
            # target
            prefix.extend(
                [
                    vocab_languages["tgt"]
                    + "_"
                    + vocab_type["tgt"]
                    + "_"
                    + str(vocab_size[1] // 1000)
                    + "k"
                    for vocab_size in vocab_sizes
                ]
            )
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

    @staticmethod
    def convert_to_parameters(
        path_corpus: Union[Dict[str, str], str],
        model_prefix: Union[List[str], str],
        vocab_size: Union[List[int], int],
        model_type: Union[Dict[str, str], str],
    ) -> Tuple[
        Union[List[str], List[Any]],
        Union[List[str], list],
        Union[List[int], List[Any]],
        Union[List[str], List[Any], Dict[str, str]],
    ]:
        def modify_vocab_size(two_dimensional_list):
            dim = np.array(two_dimensional_list).ndim
            if dim == 2:
                output = []
                for list_ in two_dimensional_list:
                    output.append(list_[0])
                for list_ in two_dimensional_list:
                    output.append(list_[1])
                return output
            return two_dimensional_list

        def modify_corpus_path_or_vocab_type(path_corpus_or_vocab_type, vocab_length):
            # 'src' then 'tgt'
            ordered_path_corpus_or_vocab_type = list()
            ordered_path_corpus_or_vocab_type.append(path_corpus_or_vocab_type["src"])
            ordered_path_corpus_or_vocab_type.append(path_corpus_or_vocab_type["tgt"])

            output = []
            for v in ordered_path_corpus_or_vocab_type:
                for _ in range(vocab_length):
                    output.append(v)
            return output

        # split_unigram_bpe.yaml
        if isinstance(path_corpus, dict):
            len_vocab = len(vocab_size)
            vocab_size = modify_vocab_size(vocab_size)
            path_corpus = modify_corpus_path_or_vocab_type(path_corpus, len_vocab)
            model_type = modify_corpus_path_or_vocab_type(model_type, len_vocab)

        # shared_bpe.yaml
        if isinstance(path_corpus, str) and isinstance(model_prefix, list):
            path_corpus = [path_corpus for i in range(len(model_prefix))]
        if isinstance(model_type, str) and isinstance(model_prefix, list):
            model_type = [model_type for i in range(len(model_prefix))]

        # vocab_baseline.yaml
        if (
            isinstance(path_corpus, str)
            and isinstance(model_prefix, str)
            and isinstance(vocab_size, int)
            and isinstance(model_type, str)
        ):
            path_corpus = [path_corpus]
            model_prefix = [model_prefix]
            vocab_size = [vocab_size]
            model_type = [model_type]

        return path_corpus, model_prefix, vocab_size, model_type

    def train(self):
        (
            self.path_corpus,
            self.prefix,
            self.config.tokenizer.vocab_size,
            self.config.tokenizer.vocab_type,
        ) = self.convert_to_parameters(
            path_corpus=self.path_corpus,
            model_prefix=self.prefix,
            vocab_size=self.config.tokenizer.vocab_size,
            model_type=self.config.tokenizer.vocab_type,
        )

        for path_corpus, prefix, vocab_size, vocab_type in zip(
            self.path_corpus,
            self.prefix,
            self.config.tokenizer.vocab_size,
            self.config.tokenizer.vocab_type,
        ):
            spm.SentencePieceTrainer.Train(
                f"--input={path_corpus} \
                                                --model_prefix={prefix} \
                                                --vocab_size={vocab_size} \
                                                --model_type={vocab_type} \
                                                --num_threads={self.config.tokenizer.args.num_threads} \
                                                --max_sentence_length={self.config.tokenizer.max_sentence_length}"
            )
            print(f"Training {prefix} tokenizer succeed")
