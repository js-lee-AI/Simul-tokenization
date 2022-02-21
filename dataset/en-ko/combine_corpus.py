import os

dir_path = "/mnt/raid6/omanma1928/projects/SiMT/SCI/simul-tokenization/dataset/en-ko/"
ko_corpus_file = "aihub_ko.txt"
en_corpus_file = "aihub_en.txt"
total_corpus_file = "aihub_total.txt"

if not os.path.exists(ko_corpus_file):
    raise FileNotFoundError('{} file is not exists.'.format(ko_corpus_file))

if not os.path.exists(en_corpus_file):
    raise FileNotFoundError('{} file is not exists.'.format(en_corpus_file))

with open(dir_path+ko_corpus_file, 'r', encoding='utf8') as f:
    ko_corpus = f.readlines()
    
with open(dir_path+en_corpus_file, 'r', encoding='utf8') as f:
    en_corpus = f.readlines()
    
total_corpus = ''.join(ko_corpus + en_corpus)

print('Combine datasets...')
with open(dir_path+total_corpus_file, 'w', encoding='utf8') as f:
    f.write(total_corpus)
print('Finish !')
