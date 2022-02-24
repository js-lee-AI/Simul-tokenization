from typing import Union, Dict, List, Tuple, Any

class ConvertParameters:
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

    @staticmethod
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

    @staticmethod
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