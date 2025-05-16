import torch
import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from MindModel.utility.config import *
import numpy as np
from MindModel.utility.logger import logger
from torch.distributions import Categorical, MultivariateNormal

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]




class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = self.config['hidden_dim']
        self.n_layers = self.config['n_layers']
        self.embedding = nn.Linear(self.config['input_dim'], self.config['embedding_dim'])
        self.rnn = nn.LSTM(self.config['embedding_dim'], self.hidden_dim, self.n_layers, dropout=self.config['dropout'])
        self.dropout = nn.Dropout(dropout)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.rnn = nn.LSTM(self.config['hidden_dim'], self.config['hidden_dim'], self.config['n_layers'])

        self.policy_head = nn.Linear(self.config['hidden_dim'], self.config['action_dim'])
        self.value_head = nn.Linear(self.config['hidden_dim'], 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])


    def forward(self):
        raise NotImplementedError
    

    def act(self, encoder_hidden):
        # encoder_hidden: [n_layers, batch_size, hidden_dim]
        batch_size = encoder_hidden.size(1)
        hidden_dim = self.config['hidden_dim']

        decoder_input = torch.zeros(1, batch_size, hidden_dim, device=encoder_hidden.device)

        h = encoder_hidden  # Use all layers
        c = torch.zeros_like(h)

        out, (h_out, c_out) = self.rnn(decoder_input, (h, c))
        h_out = h_out[-1]  # use last layer’s output

        action_logits = self.policy_head(h_out)
        value = self.value_head(h_out)

        dist = Categorical(logits=action_logits)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach(), value


    
    def set_action_std(self, new_action_std):
        if self.config['has_continuous_action_space']:
            self.action_var = torch.full((self.config['action_dim'],), new_action_std * new_action_std).to(self.device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    


    



class RLSeq2Seq(nn.Module):
    def __init__(self, encoder:Encoder, decoder:Decoder, config):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.policy_old = decoder.to(device=self.device)
        self.policy_old.load_state_dict(self.decoder.state_dict())
        
        self.config = config
        self.gamma = self.config['gamma']
        self.eps_clip = self.config['eps_clip']
        self.K_epochs = self.config['K_epochs']
        
        self.buffer = RolloutBuffer()
        self.optimizer = torch.optim.Adam([
                        {'params': self.encoder.parameters(), 'lr': self.config['lr_encoder']},
                        {'params': self.decoder.parameters(), 'lr': self.config['lr_decoder']}
                    ])
        
        self.MseLoss = nn.MSELoss()

    def evaluate(self, state, action):
        if state.dim() == 2:
            # Convert from [batch_size, obs_dim] to [seq_len=1, batch_size, obs_dim]
            state = state.unsqueeze(0)
        hidden_state, _ = self.encoder(state)
        _, action_probs, state_values = self.decoder.act(hidden_state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        #dist = Categorical(logits=self.decoder.policy_head(state_value))  # reuse logits for entropy
        #dist_entropy = dist.entropy()
        
        return action_logprobs, state_values, dist_entropy
    

    def set_action_std(self, new_action_std):
        if self.config['has_continuous_action_space']:
            self.action_std = new_action_std
            self.decoder.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    
    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.config['has_continuous_action_space']:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    
    def select_action(self, state):

        if self.config['has_continuous_action_space']:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                hidden_state, _ = self.encoder(state.unsqueeze(0).unsqueeze(0))
                action, action_logprob, state_val = self.policy_old.act(hidden_state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                hidden_state, _ = self.encoder(state.unsqueeze(0).unsqueeze(0))
                action, action_logprob, state_val = self.policy_old.act(hidden_state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()
        

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.decoder.state_dict())

        # clear buffer
        self.buffer.clear()



    def save_models(self, checkpoint="models/seq2seq"):
        os.makedirs(checkpoint, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(checkpoint, "encoder.pt"))
        torch.save(self.decoder.state_dict(), os.path.join(checkpoint, "decoder.pt"))
        logger.info(f"Models saved to {checkpoint}")

    def load_models(self, checkpoint="models/seq2seq"):
        encoder_path = os.path.join(checkpoint, "encoder.pt")
        decoder_path = os.path.join(checkpoint, "decoder.pt")

        if os.path.exists(encoder_path):
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
        else:
            logger.warning(f"Encoder checkpoint not found at {encoder_path}")

        if os.path.exists(decoder_path):
            self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device))
        else:
            logger.warning(f"Decoder checkpoint not found at {decoder_path}")

        logger.info(f"Models loaded from {checkpoint}")
