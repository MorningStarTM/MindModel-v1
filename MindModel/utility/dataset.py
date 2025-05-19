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
