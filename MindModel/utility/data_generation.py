import os
import numpy as np
import gymnasium as gym
from tqdm import trange
from datetime import datetime
from MindModel.utility.logger import logger


class DataGenerator:
    def __init__(self, env_name: str, agent=None, save_dir="data", use_random_policy=True, seed=0):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.agent = agent
        self.use_random_policy = use_random_policy or agent is None
        self.save_dir = save_dir
        self.seed = seed

        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Data will be saved to: {save_dir}")
        self.env.reset(seed=seed)

    def generate(self, num_episodes=100, max_ep_len=1000):
        data = {
            "obs": [],
            "actions": [],
            "next_obs": [],
            "rewards": [],
            "dones": [],
        }

        for episode in trange(num_episodes, desc="Generating Episodes"):
            obs, _ = self.env.reset()
            for step in range(max_ep_len):
                if self.use_random_policy:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.select_action(obs)
                    if isinstance(action, tuple):  # some agents return (action, extra)
                        action = action[0]
                    if isinstance(action, np.ndarray):
                        action = action.item() if action.shape == () else action

                next_obs, reward, done, truncated, _ = self.env.step(action)

                data["obs"].append(obs)
                data["actions"].append(action)
                data["next_obs"].append(next_obs)
                data["rewards"].append(reward)
                data["dones"].append(done or truncated)

                obs = next_obs

                if done or truncated:
                    break

        for k in data:
            data[k] = np.array(data[k])

        # Save with timestamped filename
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.env_name}_{'random' if self.use_random_policy else 'trained'}_{ts}.npy"
        filepath = os.path.join(self.save_dir, filename)
        np.save(filepath, data)
        logger.info(f"Saved dataset: {filepath}")

        return data




def load_dataset(file_path):
    try:
        data = np.load(file_path, allow_pickle=True).item()
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    print(f"✅ Loaded dataset from: {file_path}\n")

    for key in data:
        print(f"{key:>10} : shape = {np.shape(data[key])}, dtype = {data[key].dtype}")

    total_samples = len(data.get("obs", []))
    print(f"\n🔎 Total samples: {total_samples}")
    return data
