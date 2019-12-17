import torch
from tqdm import tqdm
import logging
import pickle
import nltk
from collections import Counter

class movie_embedding:
    def __init__(self, rawdata_path, seed=1357, vocab_size=50000, emb_size=300):
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.word2index = {}
        self.index2word = []
        self.word_dict = Counter()
        
        self.add('<pad>')
        self.add('<bos>')
        self.add('<eos>')
        self.add('<unk>')
        self.load_embedding(rawdata_path)

    def add(self, word):
        if not word in self.word2index:
            self.word2index[word] = len(self.index2word)
            self.index2word.append(word)
    
    def load_embedding(self, rawdata_path):
        with open(rawdata_path, "rb") as f:
            data = pickle.load(f)

        for d in data: 
            tokens = self.tokenize(d[0]) + self.tokenize(d[1])
            for word in tokens:
                self.word_dict[word] += 1

        most_common = self.word_dict.most_common()
        for i, (word, count) in enumerate(most_common):
            self.add(word)
            if len(self.word2index) >= self.vocab_size: break

        #logging.info("Embedding size: {}".format(self.vectors.size()))
    def tokenize(self, sentence):
        if sentence == None: return []
        tokens = nltk.word_tokenize(sentence)
        return tokens

    def to_index(self, word):
        word = word.lower()
        if word in self.word2index:
            return self.word2index[word] 
        else:
            return self.word2index['<unk>']

    def to_word(self, index):
        if index >= len(self.index2word):
            return 'None'
        else:
            return self.index2word[index]

    def indice_to_sentence(self, indice):
        return ' '.join(self.to_word(index) for index in indice)

    def indice_to_sentenceList(self, indice):
        return [self.to_word(index) for index in indice]

class fasttext_embedding:
    def __init__(self, rawdata_path, seed=1357):
        self.word2index = {}
        self.index2word = []
        self.vectors = []
        torch.manual_seed(seed)
        self.load_embedding(rawdata_path)

    
    def load_embedding(self, rawdata_path):
        with open(rawdata_path, encoding="utf-8") as f:
            for i, line in enumerate(tqdm(f)):
                if i == 0:          # skip header
                    continue
                if i >= 30000:
                    break

                row = line.rstrip().split(' ') # rstrip() method removes any trailing characters (default space)
                word, vector = row[0], row[1:]
                word = word.lower()
                if word not in self.word2index:
                    self.index2word.append(word)
                    self.word2index[word] = len(self.word2index)
                    vector = [float(n) for n in vector] 
                    self.vectors.append(vector)
        self.vectors = torch.tensor(self.vectors)

        if '<unk>' not in self.word2index:
            self.add('<unk>')
        if '<bos>' not in self.word2index:
            self.add('<bos>')
        if '<eos>' not in self.word2index:
            self.add('<eos>')
        if '<pad>' not in self.word2index:
            self.add('<pad>', torch.zeros(1, self.get_dim()))

        #logging.info("Embedding size: {}".format(self.vectors.size()))

    def get_dim(self):
        return len(self.vectors[0])

    def add(self, word, vector=None):
        if vector is None:
            vector = torch.empty(1,self.get_dim())
            torch.nn.init.uniform_(vector)
        vector.view(1,-1)
        self.index2word.append(word)
        self.word2index[word] = len(self.word2index)
        self.vectors = torch.cat((self.vectors, vector), 0)

    def to_index(self, word):
        word = word.lower()
        if word in self.word2index:
            return self.word2index[word] 
        else:
            return self.word2index['<unk>']

    def to_word(self, index):
        if index >= len(self.index2word):
            return 'None'
        else:
            return self.index2word[index]

    def indice_to_sentence(self, indice):
        return ' '.join(indice_to_sentenceList(indice))

    
