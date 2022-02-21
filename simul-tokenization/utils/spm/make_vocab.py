import os
import argparse
import time
import hydra

import sentencepiece as spm

class spm_trainer:    
    def __init__(
        self,
        cfg_path = './config/',
        cfg_name = 'vocab_baseline.yaml',
    ):  
        self.cfg_path_and_name = cfg_path + cfg_name
        hydra.initialize(cfg_path, cfg_name)
        
        self.configs = hydra.compose(cfg_name)
        self.path_corpus = self.configs.path.data_dir + self.configs.path.corpus_name
        
        self.prints_args(self)
        self.check_configs(self)
        
    @staticmethod
    def prints_args(self):
        self.check_path(self)
        print(f'Your config file: {self.cfg_path_and_name}')
        print(f'Your configs: ')
        print(self.configs)
        time.sleep(7)

    @staticmethod
    def check_configs(self):
        #if isinstance(self.configs.tokenizer.vocab_type, list):
        print(self.configs.tokenizer.vocab_type)
        if not self.configs.tokenizer.vocab_type in ['unigram', 'bpe', 'word', 'char']:
            raise AssertionError(f'{self.configs.tokenizer.vocab_type} is not supported.')          
        
    @staticmethod
    def check_path(self):
        if not os.path.exists(self.configs.path.data_dir):
            raise OSError('{} directory is not exists.'.format(self.configs.path.data_dir))
        
        if not os.path.exists(self.path_corpus):
            raise FileNotFoundError('{} file is not exists.'.format(self.path_corpus))
        else:
             print(f"Vocab Training Courpus File: {self.path_corpus}")
        
    def train(self):
        print('\n')
        print('Train SPM...')
        spm.SentencePieceTrainer.Train(f'--input={self.path_corpus} \
                                        --model_prefix={self.configs.tokenizer.prefix} \
                                        --vocab_size={self.configs.tokenizer.vocab_size} \
                                        --model_type={self.configs.tokenizer.vocab_type} \
                                        --num_threads={self.configs.tokenizer.args.num_threads} \
                                        --max_sentence_length=99999')
        print('Training Vocabulary Completely succeed')
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set the Hydra Configuration file')
    parser.add_argument('--cfg-name', '-c', type=str,
                        default='vocab_baseline.yaml',
                        help='hydra config file name (.yalm)')
    args = parser.parse_args()
    
    trainer = spm_trainer(cfg_path = './config/', 
                          cfg_name = args.cfg_name,
                          )
    trainer.train()
