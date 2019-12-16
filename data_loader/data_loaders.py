import torch
from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import pickle
import nltk
nltk.download('punkt')


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)



class ThesisTaggingDataLoader(BaseDataLoader):
    """
    ThesisTagging data loading demo using BaseDataLoader
    """
    def __init__(self, train_data_path, test_data_path, batch_size, num_classes, embedding, padding="<pad>", padded_len=40,
            shuffle=True, validation_split=0.0, num_workers=1, training=True):
        if training:
            data_path = train_data_path
        else:
            data_path = test_data_path
            
        self.dataset = ThesisTaggingDataset(data_path, embedding, num_classes, padding, padded_len, training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=self.dataset.collate_fn)

    


class ThesisTaggingDataset(Dataset):
    def __init__(self, data_path, embedding, num_classes, padding, padded_len, training):
        self.embedding = embedding
        self.data_path = data_path
        self.padded_len = padded_len
        self.num_classes = num_classes
        self.padding = self.embedding.to_index(padding)
        self.training = training

        with open(data_path, "rb") as f:
            data = pickle.load(f)

        self.dataset = []
        for i in data:
            self.dataset += i
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index] 
        sentence_indice = self.sentence_to_indices(data["sentence"])
        if self.training:
            return [data["number"], sentence_indice, data["label"]]
        else:
            return [data["number"], sentence_indice]

    def tokenize(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        #print(tokens)

        return tokens


    def sentence_to_indices(self, sentence):
        sentence = sentence.lower()
        sentence = self.tokenize(sentence)

        ret = []
        for word in sentence:
            ret.append(self.embedding.to_index(word))
        return ret

    def _pad_to_len(self, arr, eos=False, bos=False):
        n_special = (1 if bos else 0) + (1 if eos else 0)
        if len(arr) > self.padded_len-n_special:
            return ([self.embedding.to_index('<bos>')] if bos else []) + arr[:self.padded_len-n_special] + ([self.embedding.to_index('<eos>')] if eos else [])
        else: 
            return ([self.embedding.to_index('<bos>')] if bos else []) + arr + ([self.embedding.to_index('<eos>')] if eos else []) + [self.padding] * (self.padded_len - n_special - len(arr))

    def _to_one_hot(self, y):
        ret = [0 for i in range(self.num_classes)]
        for l in y:
            ret[l] = 1
        return ret

    def collate_fn(self, datas):
        batch = {}
        if self.training:
            batch['label'] = torch.tensor([ self._to_one_hot(r[2]) for r in datas])
        batch['sentence'] = torch.tensor([ self._pad_to_len(r[1]) for r in datas])
        batch['bos_sentence'] = torch.tensor([ self._pad_to_len(r[1], bos=True) for r in datas])
        batch['eos_sentence'] = torch.tensor([ self._pad_to_len(r[1], eos=True) for r in datas])
        batch['number'] = [ r[0] for r in datas]

        return batch


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


