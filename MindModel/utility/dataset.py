import os
import numpy as np
import torch
from torch.utils.data import Dataset

class HorizonDataset(Dataset):
    def __init__(self, npy_path: str, horizon: int = 1):
        """
        Args:
            npy_path (str): Path to the .npy file containing the dataset.
            horizon (int): Number of future steps to predict.
        """
        data = np.load(npy_path, allow_pickle=True).item()
        self.horizon = horizon

        self.obs = data['obs']
        self.actions = data['actions']
        self.next_obs = data['next_obs']
        self.rewards = data['rewards']
        self.dones = data['dones']

        self.length = len(self.obs) - horizon

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        obs = self.obs[idx]
        action_seq = self.actions[idx:idx + self.horizon]
        next_obs_seq = self.next_obs[idx:idx + self.horizon]
        reward_seq = self.rewards[idx:idx + self.horizon]
        done_seq = self.dones[idx:idx + self.horizon]

        return {
            "obs": torch.tensor(obs, dtype=torch.float32),
            "actions": torch.tensor(action_seq, dtype=torch.long),
            "next_obs": torch.tensor(next_obs_seq, dtype=torch.float32),
            "rewards": torch.tensor(reward_seq, dtype=torch.float32),
            "dones": torch.tensor(done_seq, dtype=torch.float32),
        }



class TrajectoryWindowDataset(Dataset):
    def __init__(self, obs, actions, next_obs, rewards, dones, horizon=16):
        """
        obs: [N, obs_dim]
        actions: [N,] (or [N, action_dim] if one-hot)
        next_obs: [N, obs_dim]
        rewards: [N,]
        dones: [N,]
        """
        super().__init__()
        self.obs = obs
        self.actions = actions
        self.next_obs = next_obs
        self.rewards = rewards
        self.dones = dones
        self.horizon = horizon

        self.N = len(obs)
        # Number of usable samples per episode/trajectory (leave one at end for target)
        self.max_start = self.N - self.horizon - 1

    def __len__(self):
        return self.max_start

    def __getitem__(self, idx):
        # Indices for context window
        idxs = np.arange(idx, idx + self.horizon)
        # Inputs to the decoder (context): [traj, action] pairs for steps idx ... idx+horizon-1
        traj_obs = self.obs[idxs]          # [horizon, obs_dim]
        traj_rewards = self.rewards[idxs]  # [horizon]
        traj_dones = self.dones[idxs]      # [horizon]
        traj = np.concatenate([
            traj_obs,                             # [horizon, obs_dim]
            traj_rewards[:, None],                # [horizon, 1]
            traj_dones[:, None]                   # [horizon, 1]
        ], axis=-1)                              # [horizon, obs_dim + 2]

        # Actions for each context step
        if len(self.actions.shape) == 1:  # If discrete, shape [N,]
            action_seq = np.eye(np.max(self.actions) + 1)[self.actions[idxs]]  # one-hot encode
        else:
            action_seq = self.actions[idxs]   # already one-hot or continuous

        # Targets are the next step after the window
        target_next_obs = self.next_obs[idxs]     # [horizon, obs_dim]
        target_rewards = self.rewards[idxs + 1]   # [horizon]
        target_dones = self.dones[idxs + 1]       # [horizon]
        target_actions = self.actions[idxs + 1]   # [horizon,] or [horizon, action_dim]

        return {
            "obs": self.obs[idx],  # current obs (encoder input)
            "traj_seq": traj.astype(np.float32),           # [horizon, obs_dim+2]
            "action_seq": action_seq.astype(np.float32),   # [horizon, action_dim]
            "target_next_obs": target_next_obs.astype(np.float32),  # [horizon, obs_dim]
            "target_rewards": target_rewards.astype(np.float32),    # [horizon]
            "target_dones": target_dones.astype(np.float32),        # [horizon]
            "target_actions": target_actions.astype(np.int64),      # [horizon] (int64 for CE loss)
        }
    
    
def generate_square_subsequent_mask(sz, device=None):
    # Make upper-triangular bool mask
    bool_mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
    mask = torch.zeros(sz, sz, device=device)
    mask[bool_mask] = float('-inf')
    return mask