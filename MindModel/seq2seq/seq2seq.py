import torch
import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
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
        self.prev_contexts = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.prev_contexts[:]





class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = self.config['hidden_dim']
        self.n_layers = self.config['n_layers']
        self.embedding = nn.Linear(self.config['input_dim'], self.config['embedding_dim'])
        self.rnn = nn.LSTM(self.config['embedding_dim'], self.hidden_dim, self.n_layers, dropout=self.config['dropout'])
        self.dropout = nn.Dropout(0.1)
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
        self.context_embed = nn.Linear(config['input_dim'] + config['action_dim'], config['hidden_dim'])  


        self.policy_head = nn.Linear(self.config['hidden_dim'], self.config['action_dim'])
        self.value_head = nn.Linear(self.config['hidden_dim'], 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])



    def forward(self):
        raise NotImplementedError
    

    def act(self, encoder_hidden,prev_context=None):
        batch_size = encoder_hidden.size(1)
        hidden_dim = self.config['hidden_dim']


        if prev_context is None:
            decoder_input = torch.zeros(1, batch_size, self.config['hidden_dim'], device=self.device)
        else:
            embedded_context = self.context_embed(prev_context)  # [batch_size, hidden_dim]
            decoder_input = embedded_context.unsqueeze(0)  # [1, batch_size, hidden_dim]
        # encoder_hidden: [n_layers, batch_size, hidden_dim]
        

        h = encoder_hidden  # Use all layers
        c = torch.zeros_like(h)

        out, (h_out, c_out) = self.rnn(decoder_input, (h, c))
        h_out = h_out[-1]  # use last layer’s output

        action_logits = self.policy_head(h_out)
        value = self.value_head(h_out)


        return action_logits, value


    
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

    def evaluate(self, state, action, prev_context=None):
        if state.dim() == 2:
            state = state.unsqueeze(0)
        hidden_state, _ = self.encoder(state)
        if prev_context is not None:
            prev_context = prev_context.to(self.device)
        action_probs, state_values = self.decoder.act(hidden_state, prev_context)
        dist = Categorical(logits=action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, state_values, dist_entropy


        #dist = Categorical(logits=self.decoder.policy_head(state_value))  # reuse logits for entropy
        #dist_entropy = dist.entropy()
        
        
    

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

    
    def select_action(self, state, prev_context=None):

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
                hidden_state, _ = self.encoder(state.unsqueeze(0).unsqueeze(0))  # [1, 1, obs_dim]
                if prev_context is not None:
                    prev_context = torch.FloatTensor(prev_context).to(self.device).unsqueeze(0)  # [1, context_dim]
                action_logits, state_val = self.policy_old.act(hidden_state, prev_context)
                dist = Categorical(logits=action_logits)
                action = dist.sample()
                action_logprob = dist.log_prob(action)



            self.buffer.states.append(state)
            self.buffer.actions.append(action.detach())
            self.buffer.logprobs.append(action_logprob.detach())
            self.buffer.state_values.append(state_val.detach())

            return action.item(), prev_context.squeeze(0).cpu().numpy() if prev_context is not None else None
        

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
        old_prev_contexts = torch.squeeze(torch.stack(self.buffer.prev_contexts, dim=0)).detach().to(self.device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions, old_prev_contexts)

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
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=0.5)

            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.decoder.state_dict())

        # clear buffer
        self.buffer.clear()



    def save_models(self, checkpoint=f"models/seq2seq"):
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









###########################################################################################################################################################3
class MindEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = self.config['hidden_dim']
        self.n_layers = self.config['n_layers']
        self.embedding = nn.Linear(self.config['input_dim'], self.config['embedding_dim'])
        self.rnn = nn.LSTM(self.config['embedding_dim'], self.hidden_dim, self.n_layers, dropout=self.config['dropout'])
        self.dropout = nn.Dropout(0.1)

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
    


class MindDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

        self.rnn = nn.LSTM(self.config['hidden_dim'], self.config['hidden_dim'], self.config['n_layers'])
        self.context_embed = nn.Linear(self.config['input_dim'] + self.config['action_dim'], self.config['hidden_dim'])

        # output heads
        self.next_obs_head = nn.Linear(self.config['hidden_dim'], self.config['input_dim'])
        self.reward_head = nn.Linear(self.config['hidden_dim'], 1)
        self.done_head = nn.Linear(self.config['hidden_dim'], 1)  # sigmoid for binary


    def forward_(self, hidden_state, action, prev_obs):
        # Ensure action is converted to one-hot correctly
        action_one_hot = F.one_hot(action, num_classes=self.config['action_dim']).float().to(self.device)
        logger.debug(f"Action one-hot shape: {action_one_hot.shape}")

        # Make sure both are 1D
        if prev_obs.dim() == 2:
            prev_obs = prev_obs.squeeze(0)  # From [1, input_dim] to [input_dim]
        logger.info(f"prev obs shape: {prev_obs.shape}")
        # Concatenate: [input_dim + action_dim]
        prev_cont = torch.cat([prev_obs, action_one_hot], dim=-1).unsqueeze(0)  # Shape: [1, input+action]
        
        # Embed and reshape for RNN: [1, 1, hidden_dim]
        embedded_input = self.context_embed(prev_cont).unsqueeze(0)

        # Init hidden/cell
        h = hidden_state
        c = torch.zeros_like(h).to(self.device)

        out, _ = self.rnn(embedded_input, (h, c))
        out = out.squeeze(0).squeeze(0)

        # Heads
        next_obs = self.next_obs_head(out)
        reward = self.reward_head(out)
        done = torch.sigmoid(self.done_head(out))

        return next_obs, reward, done
    
    def forward(self, hidden_state, action, prev_obs):
        """
        hidden_state: [n_layers, batch_size, hidden_dim]
        action: [batch_size]  (discrete action index)
        obs: [batch_size, input_dim]
        """
        assert torch.max(action).item() < self.config['action_dim'], f"Invalid action index {torch.max(action).item()}, must be < action_dim={self.config['action_dim']}"

        action_one_hot = F.one_hot(action, num_classes=self.config['action_dim']).float().to(self.device)
        prev_cont = torch.cat([prev_obs, action_one_hot], dim=-1)  # [batch_size, input+action]
        embedded_input = self.context_embed(prev_cont).unsqueeze(0)  # [1, batch_size, hidden_dim]

        h = hidden_state  # [n_layers, batch_size, hidden_dim]
        c = torch.zeros_like(h).to(self.device)

        out, _ = self.rnn(embedded_input, (h, c))  # [1, batch, hidden_dim]
        out = out.squeeze(0)  # [batch, hidden_dim]

        next_obs = self.next_obs_head(out)
        reward = self.reward_head(out)
        done = torch.sigmoid(self.done_head(out))

        return next_obs, reward, done



    

    
class MindModel(nn.Module):
    def __init__(self, encoder:MindEncoder, decoder:MindDecoder, config):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.config = config
        self.horizon = self.config['horizon']
        
        self.optimizer = torch.optim.Adam([
                        {'params': self.encoder.parameters(), 'lr': self.config['lr_encoder']},
                        {'params': self.decoder.parameters(), 'lr': self.config['lr_decoder']}
                    ])

    
    def forward(self, obs, action, prev_obs):
        hidden_state, cell = self.encoder(obs)
        next_obs, reward, done = self.decoder(hidden_state, action, prev_obs)
        return next_obs, reward, done
    


    def horizon_predict(self, obs, actions, prev_obs):
        """
        obs:        [batch_size, 1, input_dim] or [1, 1, input_dim]
        actions:    [batch_size, horizon]
        prev_obs:   [batch_size, input_dim]
        """
        hidden_state, _ = self.encoder(obs)  # [n_layers, batch_size, hidden_dim]

        pred_next_obs = []
        pred_reward = []
        pred_done = []

        for step in range(self.horizon):
            current_actions = actions[:, step]  # [batch_size]
            next_obs, reward, done = self.decoder(hidden_state, current_actions, prev_obs)

            pred_next_obs.append(next_obs)  # each: [batch_size, input_dim]
            pred_reward.append(reward)      # each: [batch_size, 1]
            pred_done.append(done)          # each: [batch_size, 1]

            prev_obs = next_obs.detach()  # for next step

        return {
            "next_obs": pred_next_obs,  # list of Tensors [B, input_dim]
            "reward": pred_reward,      # list of Tensors [B, 1]
            "done": pred_done           # list of Tensors [B, 1]
        }
    

    def save(self, folder: str = "checkpoints", filename: str = "mindmodel.pt"):
        
        save_path = os.path.join(folder, filename)

        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)

        logger.info(f"✅ Model saved at: {save_path}")


    def load(self, folder: str = "checkpoints", filename: str = "mindmodel.pt"):
        load_path = os.path.join(folder, filename)

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"❌ Model file not found at: {load_path}")

        checkpoint = torch.load(load_path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info(f"✅ Model loaded from: {load_path}")






###############################################################################################################################################################
###############################################################################################################################################################
###############################################################################################################################################################


class MindPolicyDecoder(MindDecoder):
    def __init__(self, config):
        super().__init__(config)
        # Replace prediction heads with action policy head
        self.policy_head = nn.Linear(self.config['hidden_dim'], self.config['action_dim'])
        self.value_head = nn.Linear(self.config['hidden_dim'], 1)
        # Optionally, delete the old heads (cleaner)
        del self.next_obs_head
        del self.reward_head
        del self.done_head

    def act(self, encoder_hidden,prev_context=None):
        batch_size = encoder_hidden.size(1)


        if prev_context is None:
            decoder_input = torch.zeros(1, batch_size, self.config['hidden_dim'], device=self.device)
        else:
            embedded_context = self.context_embed(prev_context)  # [batch_size, hidden_dim]
            decoder_input = embedded_context.unsqueeze(0)  # [1, batch_size, hidden_dim]
        # encoder_hidden: [n_layers, batch_size, hidden_dim]
        

        h = encoder_hidden  # Use all layers
        c = torch.zeros_like(h)

        out, (h_out, c_out) = self.rnn(decoder_input, (h, c))
        h_out = h_out[-1]  # use last layer’s output

        action_logits = self.policy_head(h_out)
        value = self.value_head(h_out)


        return action_logits, value
    
    def set_action_std(self, new_action_std):
        if self.config['has_continuous_action_space']:
            self.action_var = torch.full((self.config['action_dim'],), new_action_std * new_action_std).to(self.device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")





class MindModelAgent(nn.Module):
    def __init__(self, config, pretrained_model_path, pretrained_model_name="mindmodel.pt"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.encoder = MindEncoder(config).to(self.device)
        self.decoder = MindPolicyDecoder(config).to(self.device)
        self.policy_old = MindPolicyDecoder(config).to(self.device)
        self.buffer = RolloutBuffer()
        
        self.gamma = self.config['gamma']
        self.eps_clip = self.config['eps_clip']
        self.K_epochs = self.config['K_epochs']
        
        # Load the trained weights (MindModel was trained with MindDecoder, not policy decoder)
        pretrained = MindModel(self.encoder, MindDecoder(config), config)
        pretrained.load(pretrained_model_path, pretrained_model_name)
        self.encoder.load_state_dict(pretrained.encoder.state_dict())
        # For the decoder, load only shared weights (not output heads)
        decoder_state_dict = pretrained.decoder.state_dict()
        policy_decoder_dict = self.decoder.state_dict()
        # Only load matching keys (shared RNN/context_embed weights)
        filtered_dict = {k: v for k, v in decoder_state_dict.items() if k in policy_decoder_dict}
        policy_decoder_dict.update(filtered_dict)
        self.decoder.load_state_dict(policy_decoder_dict)
        self.policy_old.load_state_dict(self.decoder.state_dict())

        self.optimizer = torch.optim.Adam([
                        {'params': self.encoder.parameters(), 'lr': self.config['lr_encoder']},
                        {'params': self.decoder.parameters(), 'lr': self.config['lr_decoder']}
                    ])

    def forward(self, obs, prev_obs, prev_action):
        # obs: [batch, 1, input_dim]
        # prev_obs: [batch, input_dim]
        # prev_action: [batch]
        hidden_state, _ = self.encoder(obs)
        logits = self.decoder(hidden_state, prev_obs, prev_action)
        return logits

    def select_action(self, state, prev_context=None):

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
                hidden_state, _ = self.encoder(state.unsqueeze(0).unsqueeze(0))  # [1, 1, obs_dim]
                if prev_context is not None:
                    prev_context = torch.FloatTensor(prev_context).to(self.device).unsqueeze(0)  # [1, context_dim]
                action_logits, state_val = self.policy_old.act(hidden_state, prev_context)
                dist = Categorical(logits=action_logits)
                action = dist.sample()
                action_logprob = dist.log_prob(action)



            self.buffer.states.append(state)
            self.buffer.actions.append(action.detach())
            self.buffer.logprobs.append(action_logprob.detach())
            self.buffer.state_values.append(state_val.detach())

            return action.item(), prev_context.squeeze(0).cpu().numpy() if prev_context is not None else None


    
    def evaluate(self, state, action, prev_context=None):
        if state.dim() == 2:
            state = state.unsqueeze(0)
        hidden_state, _ = self.encoder(state)
        if prev_context is not None:
            prev_context = prev_context.to(self.device)
        action_probs, state_values = self.decoder.act(hidden_state, prev_context)
        dist = Categorical(logits=action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

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
        old_prev_contexts = torch.squeeze(torch.stack(self.buffer.prev_contexts, dim=0)).detach().to(self.device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions, old_prev_contexts)

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
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=0.5)

            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.decoder.state_dict())

        # clear buffer
        self.buffer.clear()

    
    def save_models(self, checkpoint="models/seq2seq"):
        os.makedirs(checkpoint, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(checkpoint, f"encoder_{self.config['horizon']}.pt"))
        torch.save(self.decoder.state_dict(), os.path.join(checkpoint, f"decoder_{self.config['horizon']}.pt"))
        logger.info(f"Models saved to {checkpoint}")

    def load_models(self, checkpoint="models/seq2seq"):
        encoder_path = os.path.join(checkpoint, f"encoder_{self.config['horizon']}.pt")
        decoder_path = os.path.join(checkpoint, f"decoder_{self.config['horizon']}.pt")

        if os.path.exists(encoder_path):
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
        else:
            logger.warning(f"Encoder checkpoint not found at {encoder_path}")

        if os.path.exists(decoder_path):
            self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device))
        else:
            logger.warning(f"Decoder checkpoint not found at {decoder_path}")

        logger.info(f"Models loaded from {checkpoint}")