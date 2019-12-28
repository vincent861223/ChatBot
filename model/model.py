import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from base import BaseModel


class Encoder(nn.Module):
    def __init__(self, embedding, hidden_size, rnn_cell='GRU', bidirectional=False, n_layers=1, dropout=0.0, device='cpu'):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = rnn_cell
        #self.padding_idx = self.embedding.to_index(padding)
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device
        self.n_init = (2 if bidirectional == True else 1) * n_layers
        self.vocab_size = embedding.vocab_size
        self.emb_size = embedding.emb_size

        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        #self.embedding.weight = nn.Parameter(embedding.vectors)
        if rnn_cell == 'GRU': self.rnn = nn.GRU(self.emb_size, self.hidden_size, batch_first=True, dropout=self.dropout, num_layers=self.n_layers)
        #self.freeze_embedding()

    def freeze_embedding(self):
        for param in self.embedding.parameters():
            param.requires_grad = False

    def forward(self, source):
        # source: (batch, seq_len)
        #init_hidden = torch.randn(self.n_init, source.size(0), self.hidden_size).to(self.device) #(n_layer*n_direction, batch, hidden_size)
        source = self.embedding(source) # (batch, seq_len) -> (batch, seq_len, emb_size)
        output, hidden = self.rnn(source, None) #(batch, seq_len, emb_size) -> (batch, seq_len, emb_size*n_direction), (n_layer*n_direction, batch, hidden_size)
        return output, hidden #(n_layer*n_direction, batch, hidden_size)


class Decoder(nn.Module):
    def __init__(self, embedding, hidden_size, rnn_cell='GRU', bidirectional=False, n_layers=1, dropout=0.2, device='cpu', teaching_force_rate=0.0, use_attn=False, method=None, padded_len=None):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = rnn_cell
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device
        self.n_init = (2 if bidirectional == True else 1) * n_layers
        self.teaching_force_rate = teaching_force_rate
        self.vocab_size = embedding.vocab_size
        self.emb_size = embedding.emb_size
        self.use_attn = use_attn
        self.step = 0

        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=-1)
        self.embedding_org = embedding
        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        #self.embedding.weight = nn.Parameter(embedding.vectors)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        if rnn_cell == 'GRU': self.rnn = nn.GRU(self.emb_size, self.hidden_size, batch_first=True, dropout=self.dropout, num_layers=self.n_layers)
        if self.use_attn:
            self.attn = Attention(hidden_size=self.hidden_size, method=method, padded_len=padded_len)
        #self.freeze_embedding()

    def freeze_embedding(self):
        for param in self.embedding.parameters():
            param.requires_grad = False

    def forward(self, label, init_hidden, encoder_output=None):
        if(self.step > 2000): self.teaching_force_rate = 0.2
        self.step += 1
        use_teaching_force = True if random.random() <= self.teaching_force_rate else False
        # source: (batch, seq_len)
        #input = self.relu(self.embedding(input)) # (batch, seq_len) -> (batch, seq_len, emb_size)
        batch, seq_len = label.size(0), label.size(1)
        outputs = []

        hidden = init_hidden

        if use_teaching_force:
            for i in range(seq_len):
                input = label[:, i].unsqueeze(1)
                #print(label)
                #print(str(i) + ': ' + self.embedding_org.indice_to_sentence(label[0].tolist()))
                input = self.relu(self.embedding(input))
                if self.use_attn: 
                    attn_output = self.attn(encoder_output, input, hidden)
                    output, hidden = self.rnn(attn_output, hidden)
                else: 
                    output, hidden = self.rnn(input, hidden)
                output = self.softmax(self.linear(output))
                last_predict = output.max(2)[1]
                #print(str(i) + ': ' + self.embedding_org.indice_to_sentence(last_predict[0].tolist()))
                outputs.append(output)

        else: 
            input = label[:, 0].unsqueeze(1)
            input = self.relu(self.embedding(input))
            for i in range(seq_len):
                if self.use_attn: 
                    attn_output = self.attn(encoder_output, input, hidden)
                    output, hidden = self.rnn(attn_output, hidden)
                else: 
                    output, hidden = self.rnn(input, hidden)
                output = self.softmax(self.linear(output))
                outputs.append(output)
                last_predict = output.max(2)[1]
                input = self.relu(self.embedding(last_predict))

        outputs = torch.cat(outputs, dim=1)
        return outputs

class Attention(nn.Module):
    def __init__(self, hidden_size, method, padded_len):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.method = method
        self.attn = nn.Linear(hidden_size*2, padded_len)
        self.attn_combine = nn.Linear(hidden_size*2, hidden_size)
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
    def forward(self, encoder_output, decoder_input_embeded, decoder_hidden):
        # encoder_output: [batch, seq_len, hidden_size=embedded_size]
        # decoder_input_embeded: [batch, 1, embedded_size]
        # decoder_hidden: [batch, 1, embedded_size]
        decoder_hidden = decoder_hidden.permute(1, 0, 2)
        # print(encoder_output.size())
        # print(decoder_input_embeded.size())
        # print(decoder_hidden.size())
        similarity = self.attn(torch.cat((decoder_input_embeded, decoder_hidden), dim=-1))
        attn_weight = self.softmax(similarity) # [batch, 1, padded_len]

        attn_applied = torch.bmm(attn_weight, encoder_output) #[batch, 1, hidden_size]
        output = self.relu(self.attn_combine(torch.cat((attn_applied, decoder_input_embeded), dim=-1)))
        return output 

        

class ChatBotModel(BaseModel):
    def __init__(self, embedding, hidden_size, rnn_cell='GRU', bidirectional=False, n_layers=1, dropout=0.2, device='cpu', teaching_force_rate=0.0, use_attn=False, method='concat', padded_len=10):
        super().__init__()
        self.use_attn = use_attn
        self.embedding = embedding
        if self.use_attn: 
            self.hidden_size = embedding.emb_size
        else:
            self.hidden_size = hidden_size
        self.encoder = Encoder(self.embedding, self.hidden_size, rnn_cell=rnn_cell, bidirectional=bidirectional, n_layers=n_layers, dropout=dropout, device=device)
        self.decoder = Decoder(self.embedding, self.hidden_size, rnn_cell=rnn_cell, bidirectional=bidirectional, n_layers=n_layers, dropout=dropout, device=device, teaching_force_rate=teaching_force_rate, use_attn=self.use_attn, method=method, padded_len=padded_len)

    def forward(self, source, target):
        # print('> : ' + self.embedding.indice_to_sentence(source[0].tolist()))
        # print('= : ' + self.embedding.indice_to_sentence(target[0].tolist()))
        encoder_output, encoder_hidden = self.encoder(source)
        if self.use_attn:
            output = self.decoder(target, encoder_hidden, encoder_output)
        else:
            output = self.decoder(target, encoder_hidden)

        return output


