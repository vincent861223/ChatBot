import torch
from base import BaseDataLoader
from torch.utils.data import Dataset
import pickle
import nltk
nltk.download('punkt')

class SentencePairDataloader(BaseDataLoader):
    def __init__(self, train_data_path, test_data_path, batch_size, embedding, padding='<pad>', padded_len=40,
            shuffle=True, validation_split=0.0, num_workers=1, training=True, add_bos=False, add_eos=False):
        data_path = train_data_path if training else test_data_path
        self.embedding = embedding
        self.dataset = SentencePairDataset(data_path, embedding, padding, padded_len, add_bos, add_eos)
        print(self.dataset)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=self.dataset.collate_fn)


class SentencePairDataset(Dataset):
    def __init__(self, data_path, embedding, padding, padded_len, add_bos=False, add_eos=False):
        self.data_path = data_path
        self.embedding = embedding
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.padded_len = padded_len
        self.padding = self.embedding.to_index(padding)
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        self.data = [d for d in data] # [['Run.', '你用跑的。'], ['Wait!', '等等！'] ... ]

    def __str__(self):
        ret = ''
        for i in range(5):
            ret += '{}, {}\n'.format(self[i]['org'], self[i]['indexed'])
        return ret

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index] # ['Run.', '你用跑的。']
        indexed_sent = [self.sentence_to_indice(data[0]), self.sentence_to_indice(data[1])]
        indexed = [self._pad_to_len(indexed_sent[0]), self._pad_to_len(indexed_sent[1])]
        indexed_bos = [self._pad_to_len(indexed_sent[0], add_bos=True), self._pad_to_len(indexed_sent[1], add_bos=True)]
        indexed_eos = [self._pad_to_len(indexed_sent[0], add_eos=True), self._pad_to_len(indexed_sent[1], add_eos=True)]

        return {'org': data, 'indexed': indexed, 'indexed_bos': indexed_bos, 'indexed_eos': indexed_eos}

    def tokenize(self, sentence):
        if(not isinstance(sentence, str)): 
            sentence = ''
        tokens = nltk.word_tokenize(sentence) # '你用跑的。' -> ['你', '用', '跑', '的', '。']
        return tokens

    def sentence_to_indice(self, sentence):
        # input: sentence(str)
        # output: indexed_sentence(list)
        tokens = self.tokenize(sentence)
        indexed_sentence = [self.embedding.to_index(token) for token in tokens]
        return indexed_sentence

    def _pad_to_len(self, arr, add_bos=False, add_eos=False):
        n_special = (1 if add_bos else 0) + (1 if add_eos else 0)
        if len(arr) > self.padded_len-n_special:
            return ([self.embedding.to_index('<bos>')] if add_bos else []) + arr[:self.padded_len-n_special] + ([self.embedding.to_index('<eos>')] if add_eos else [])
        else: 
            return ([self.embedding.to_index('<bos>')] if add_bos else []) + arr + ([self.embedding.to_index('<eos>')] if add_eos else []) + [self.padding] * (self.padded_len - n_special - len(arr))


    def collate_fn(self, batch):
        source = torch.tensor([b['indexed'][0] for b in batch])
        target = torch.tensor([b['indexed'][1] for b in batch])
        target_bos = torch.tensor([b['indexed_bos'][1] for b in batch])
        target_eos = torch.tensor([b['indexed_eos'][1] for b in batch])
        return {'source': source, 'target': target, 'target_bos': target_bos, 'target_eos': target_eos}


