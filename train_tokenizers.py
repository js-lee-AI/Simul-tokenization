from utils.train_spm_tokenizer import SPMTrainer

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

    trainer = SPMTrainer(
        cfg_path="../config",
        cfg_name=args.cfg_name,
    )
    trainer.train()
