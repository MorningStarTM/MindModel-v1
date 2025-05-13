import torch
import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from MindModel.utility.config import *
import numpy as np

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.next_states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.next_states), np.array(self.actions), np.array(self.probs),np.array(self.vals),np.array(self.rewards), np.array(self.dones), batches
    
    def store_memory(self, state, next_state, action, probs, vals, reward, done):
        self.states.append(state)
        self.next_states.append(next_state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.next_states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []




class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = self.config['hidden_dim']
        self.n_layers = self.config['n_layers']
        self.embedding = nn.Linear(self.config['input_dim'], self.config['embedding_dim'])
        self.rnn = nn.LSTM(self.config['embedding_dim'], self.hidden_dim, self.n_layers, dropout=self.config['dropout'])
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
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rnn = nn.LSTM(self.config['hidden_dim'], self.config['hidden_dim'])
        self.policy_head = nn.Linear(self.config['hidden_dim'], self.config['action_dim'])
        self.value_head = nn.Linear(self.config['hidden_dim'], 1)

    def forward(self, encoder_hidden):
        # decoder_input could be zeros or learned start token
        batch_size = encoder_hidden.size(1)
        decoder_input = torch.zeros(batch_size, self.config['hidden_dim']).to(encoder_hidden.device)

        h, c = encoder_hidden[-1], torch.zeros_like(encoder_hidden[-1])  # Use last encoder layer
        h, c = self.rnn(decoder_input.unsqueeze(0), (h.unsqueeze(0), c.unsqueeze(0)))  # Match LSTM input shape
        h = h.squeeze(0)

        action_logits = self.policy_head(h)
        value = self.value_head(h)
        return action_logits, value
    



class RLSeq2Seq(nn.Module):
    def __init__(self, encoder:Encoder, decoder:Decoder, config):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        
        self.gamma = config['gamma']
        self.policy_clip = config['policy_clip']
        self.n_epochs = config['n_epochs']
        self.gae_lambda = config['gae_lambda']
        self.name = "Seq2Seq-Agent"
        self.alpha = config['alpha']
        self.memory = PPOMemory(config['batch_size'])

    
    def remember(self, state, next_state, action, probs, vals, reward, done):
        self.memory.store_memory(state, next_state, action, probs, vals, reward, done)

    def choose_action(self, obs):
        # obs = [seq_len, batch_size, obs_dim]
        state = torch.from_numpy(obs).float().to(self.device)
        encoder_hidden, _ = self.encoder(state)
        action_logits, value = self.decoder(encoder_hidden)

        action_dist = torch.distributions.Categorical(logits=action_logits)
        action = action_dist.sample()  # shape: [batch_size]

        probs = torch.squeeze(action_dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value

    

