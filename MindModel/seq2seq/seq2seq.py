import torch
import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from MindModel.utility.config import *



class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = input_dim
        self.n_layers = n_layers
        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src length, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src length, batch size, embedding dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [src length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # outputs are always from the top hidden layer
        return hidden, cell
    


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_hidden):
        # decoder_input could be zeros or learned start token
        batch_size = encoder_hidden.size(1)
        decoder_input = torch.zeros(batch_size, hidden_dim).to(encoder_hidden.device)

        h, c = encoder_hidden[-1], torch.zeros_like(encoder_hidden[-1])  # Use last encoder layer
        h, c = self.rnn(decoder_input, (h, c))

        action_logits = self.policy_head(h)
        value = self.value_head(h)
        return action_logits, value
    



class RLSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, obs):
        # obs = [seq_len, batch_size, obs_dim]
        encoder_hidden, _ = self.encoder(obs)
        action_logits, value = self.decoder(encoder_hidden)
        return action_logits, value
