from MindModel.utility.data_generation import DataGenerator
from MindModel.utility.data_generation import load_dataset
from MindModel.baseline.ppo import PPO
from MindModel.seq2seq.seq2seq import RLSeq2Seq, Encoder, Decoder
from MindModel.seq2seq.config import config
import gymnasium as gym

env = gym.make("CartPole-v1")

config['input_dim'] = env.observation_space.shape[0]
config['action_dim'] = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]


encoder = Encoder(config)
decoder = Decoder(config)
agent = RLSeq2Seq(encoder=encoder, decoder=decoder, config=config)
agent.load_models("Models\\CartPole-v1")

gen_trained = DataGenerator(env_name="CartPole-v1", agent=agent, use_random_policy=False)
gen_trained.generate(num_episodes=100000)
