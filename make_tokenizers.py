import os
import argparse
import time
import hydra

import sentencepiece as spm
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig

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
        self.multiple_vocabsize = False
       
        self.prints_args()
        self.check_configs()
        
        # if self.multiple_vocabsize == True:
        #     self.prefix = [cfg_name[:-5] + '_' + str(vocab_size//1000) + 'k' 
        #                    for vocab_size in self.configs.tokenizer.vocab_size]
        # else:
        #     self.prefix = cfg_name[:-5] + '_' + str(self.configs.tokenizer.vocab_size//1000) + 'k' 

        self.prefix = set_tokenizer_prefix(
            multiple_flag = self.multiple_vocabsize,
            vocab_sizes = self.configs.tokenizers.vocab_size,
            cfg_name = cfg_name,
        )
     
    def prints_args(self):
        self.check_and_move_path()
        print(f'Your config file: {self.cfg_path_and_name}')
        print(f'Your configs: ')
        print(self.configs)
        time.sleep(7)

    def check_configs(self):
        print(self.configs.tokenizer.vocab_size, type(self.configs.tokenizer.vocab_size))
        if isinstance(self.configs.tokenizer.vocab_size, ListConfig):
            self.multiple_vocabsize = True
            
        if not self.configs.tokenizer.vocab_type in ['unigram', 'bpe', 'word', 'char']:
            raise AssertionError(f'{self.configs.tokenizer.vocab_type} is not supported.')          
        
    def check_and_move_path(self):
        if not os.path.exists(self.configs.path.data_dir):
            raise OSError('{} directory is not exists.'.format(self.configs.path.data_dir))
        
        if not os.path.exists(self.path_corpus):
            raise FileNotFoundError('{} file is not exists.'.format(self.path_corpus))
        else:
             print(f"Vocab Training Courpus File: {self.path_corpus}")
        
        # get abstract paths
        self.path_corpus = os.path.abspath(self.path_corpus)
        
        # make directory then move output path 
        if not os.path.exists(os.path.abspath(self.configs.path.output_path)):
            os.mkdir(os.path.abspath(self.configs.path.output_path))
        else:
            os.chdir(os.path.abspath(self.configs.path.output_path))
    
    @staticmethod
    def set_tokenizer_prefix(
        multiple_flag,
        vocab_sizes,
        cfg_name,
    ):
        # multiple vocab size
        if multiple_flag == True:
            return [cfg_name[:-5] + '_' + str(vocab_size//1000) + 'k' 
                           for vocab_size in vocab_sizes]
        # single vocab size
        else:
            return cfg_name[:-5] + '_' + str(self.configs.tokenizer.vocab_size//1000) + 'k' 
        
            
    def train(self):
        print('\n')
        # training multiple tokenizers
        if self.multiple_vocabsize == True:
            print('Train SPM...')
            for idx, prefix in enumerate(self.prefix):
                spm.SentencePieceTrainer.Train(f'--input={self.path_corpus} \
                                                --model_prefix={prefix} \
                                                --vocab_size={self.configs.tokenizer.vocab_size[idx]} \
                                                --model_type={self.configs.tokenizer.vocab_type} \
                                                --num_threads={self.configs.tokenizer.args.num_threads} \
                                                --max_sentence_length=99999')
                print(f'Training {self.prefix} Tokenizer succeed')
            print('Training All of Tokenizers Completely succeed')
        # training single tokenizer
        else:
            print('Train SPM...')
            spm.SentencePieceTrainer.Train(f'--input={self.path_corpus} \
                                --model_prefix={self.configs.tokenizer.prefix} \
                                --vocab_size={self.configs.tokenizer.vocab_size} \
                                --model_type={self.configs.tokenizer.vocab_type} \
                                --num_threads={self.configs.tokenizer.args.num_threads} \
                                --max_sentence_length=99999')
            print('Training Tokenizer Completely succeed')
            print('Finish...')
            
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
