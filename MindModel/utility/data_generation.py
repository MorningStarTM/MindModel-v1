import os
import numpy as np
import gymnasium as gym
from tqdm import trange
from datetime import datetime
from MindModel.utility.logger import logger
import cv2


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



class DataGenerator:
    def __init__(self, env_name: str, agent=None, save_dir="data", use_random_policy=True, seed=0, render_mode="rgb_array"):
        self.env_name = env_name
        self.env = gym.make(env_name, render_mode=render_mode)
        self.agent = agent
        self.use_random_policy = use_random_policy or agent is None
        self.save_dir = save_dir
        self.seed = seed

        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Data will be saved to: {save_dir}")
        self.env.reset(seed=seed)

    def resize_obs(self, obs):
        # obs: (H,W,3) np.uint8 or float
        resized = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        if resized.ndim == 2:  # if gray, expand to (H,W,1)
            resized = np.expand_dims(resized, axis=-1)
        if resized.shape[2] == 1:  # convert to RGB if needed
            resized = np.repeat(resized, 3, axis=-1)
        return resized.astype(np.uint8)  # always save as uint8

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
                frame = self.resize_obs(obs)

                if self.use_random_policy:
                    action = self.env.action_space.sample()

                else:
                    action = self.agent.select_action(frame)
                    if isinstance(action, tuple):
                        action = action[0]
                    if isinstance(action, np.ndarray):
                        action = action.item() if action.shape == () else action

                if isinstance(action, (float, int)):
                    action = np.array([action]*self.env.action_space.shape[0], dtype=np.float32)
                elif isinstance(action, list):
                    action = np.array(action, dtype=np.float32)
                elif isinstance(action, np.ndarray):
                    action = action.astype(np.float32)
                else:
                    raise ValueError(f"Unexpected action type: {type(action)}, value: {action}")

                action = np.asarray(action, dtype=np.float32)
                if action.shape != self.env.action_space.shape:
                    action = np.reshape(action, self.env.action_space.shape)
                # Always convert to native Python floats before passing to env.step (Box2D workaround)
                action_for_env = [float(x) for x in action]
                next_obs, reward, done, truncated, _ = self.env.step(action_for_env)

                next_frame = self.resize_obs(next_obs)

                data["obs"].append(frame)
                data["actions"].append(action)
                data["next_obs"].append(next_frame)
                data["rewards"].append(reward)
                data["dones"].append(done or truncated)

                obs = next_obs

                if done or truncated:
                    break

        # Convert to arrays
        for k in ["obs", "next_obs"]:
            data[k] = np.stack(data[k], axis=0)  # [N, 84, 84, 3]
        for k in ["actions", "rewards", "dones"]:
            data[k] = np.array(data[k])

        # Save
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.env_name}_{'random' if self.use_random_policy else 'trained'}_{ts}_VISUAL84.npy"
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
