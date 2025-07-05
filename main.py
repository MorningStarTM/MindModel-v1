from MindModel.seq2seq.seq2seq import RLSeq2Seq, Encoder, Decoder, MindModelAgent
from MindModel.seq2seq.trainer import Seq2SeqTrainer
from MindModel.utility.logger import logger
from MindModel.baseline.ppo import PPO
from MindModel.baseline.trainer import Trainer
import os
import gymnasium as gym
import gymnasium_robotics
from MindModel.seq2seq.config import mindmodel_agent_config
from MindModel.baseline.config import config

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

env_name = ["CartPole-v1", "LunarLander-v2", "Acrobot-v1", "MountainCar-v0"]
gymnasium_env = ['Ant-v5', "HalfCheetah-v5", "Hopper-v5", "HumanoidStandup-v5", "Humanoid-v5", "InvertedDoublePendulum-v5", "InvertedPendulum-v5", "Reacher-v5", "Walker2d-v5"]
env = gym.make(env_name[1])


config['state_dim'] = env.observation_space.shape[0]
config['action_dim'] = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
config['horizon'] = 8
config['mlp_add'] = True
agent = MindModelAgent(config, pretrained_model_path="MindModel_version\\CartPole-v1", pretrained_model_name="mindmodel_h8_20250528_094646.pt")
trainer = Seq2SeqTrainer(agent, env_name[0], config)
trainer.train()

"""
encoder = Encoder(config)
decoder = Decoder(config)
agent = RLSeq2Seq(encoder=encoder, decoder=decoder, config=config)
trainer = Seq2SeqTrainer(agent, env_name[1], config)
trainer.train()"""



"""agent = PPO(config)
trainer = Trainer(agent, env_name[1], config)
trainer.train()"""
