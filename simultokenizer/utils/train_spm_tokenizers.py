from simultokenizer.utils.train_spm_tokenizer import SPMTrainer
from simultokenizer.utils.morpheme_tokenizer import MorphemeTrainer
from simultokenizer.utils.module import ConfigModule

import argparse

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

    vocab_type = ConfigModule.return_type_of_vocab(
        cfg_path="../config",
        cfg_name=args.cfg_name,
    )

    if vocab_type == "morpheme_aware_BPE":
        trainer = MorphemeTrainer(
            cfg_path="../config",
            cfg_name=args.cfg_name,
        )
    elif vocab_type in ["unigram", "bpe", "char", "word"]:
        trainer = SPMTrainer(
            cfg_path="../config",
            cfg_name=args.cfg_name,
        )

    trainer.train()
