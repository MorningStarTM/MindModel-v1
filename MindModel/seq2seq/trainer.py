
import matplotlib.pyplot as plt
import time
import numpy as np
from collections import deque
import torch
import gymnasium as gym
from datetime import datetime
import os
import itertools
import torch
from tqdm import tqdm
from MindModel.utility.logger import logger
from MindModel.seq2seq.seq2seq import RLSeq2Seq, MindModel
import torch.nn as nn
from tqdm import trange
import numpy as np
import csv



os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Seq2SeqTrainer:
    def __init__(self, agent:RLSeq2Seq, env_name:str, config):
        
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.agent = agent
        self.best_score = 0.0
        self.score_history = []
        self.config = config
        self.episode_rewards = []  # Stores total reward per episode
        self.step_rewards = []     # Stores every single reward at each timestep

        self.log_dir = "model_logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.log_dir = self.log_dir + '/' + env_name + '/'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        run_num = 0
        current_num_files = next(os.walk(self.log_dir))[2]
        run_num = len(current_num_files)

        #### create new log file for each run
        self.log_f_name = self.log_dir + '/Seq2SeqRL_' + env_name + "_log_" + str(run_num) + ".csv"

        logger.info("current logging run number for " + env_name + " : ", run_num)
        logger.info("logging at : " + self.log_f_name)

        self.directory = "Models"
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.directory = self.directory + '/' + env_name + '/'
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.reward_folder = 'rewards'
        if not os.path.exists(self.reward_folder):
            os.makedirs(self.reward_folder)

        self.reward_folder = self.reward_folder + '/' + env_name + '/'
        if not os.path.exists(self.reward_folder):
            os.makedirs(self.reward_folder)
        

        self.trace_log_file = os.path.join(self.log_dir, f"trace_log_{env_name}.csv")
        with open(self.trace_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "step",
                "encoder_input",         # state
                "action",                # selected action
                "prev_obs",              # previous observation
                "prev_action",           # previous action
                "decoder_input"          # combined [prev_obs + one_hot(prev_action)]
            ])

    

    def train(self):
        start_time = datetime.now().replace(microsecond=0)
        logger.info("Started training at (GMT) : ", start_time)

        print("============================================================================================")

        # logging file
        log_f = open(self.log_f_name,"w+")
        log_f.write('episode,timestep,reward\n')

        # printing and logging variables
        print_running_reward = 0
        print_running_episodes = 0

        log_running_reward = 0
        log_running_episodes = 0

        time_step = 0
        i_episode = 0
        
        while time_step <= self.config['max_training_timesteps']:

            state, _ = self.env.reset()
            current_ep_reward = 0
            prev_obs = None
            prev_action = None


            for t in range(1, self.config['max_ep_len']+1):
                if prev_obs is not None and prev_action is not None:
                    prev_context = np.concatenate([prev_obs, np.eye(self.config['action_dim'])[prev_action]])
                else:
                    prev_context = None

                # select action with policy

                action, new_context = self.agent.select_action(state, prev_context=prev_context)

                next_state, reward, done, _, _ = self.env.step(action)
                self.step_rewards.append(reward)

                decoder_input = None
                if prev_obs is not None and prev_action is not None:
                    one_hot_action = np.eye(self.config['action_dim'])[prev_action]
                    decoder_input = np.concatenate([prev_obs, one_hot_action])
                else:
                    decoder_input = np.zeros(self.config['input_dim'] + self.config['action_dim'])

                with open(self.trace_log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        i_episode,
                        t,
                        state.tolist(),                 # encoder input
                        action,
                        prev_obs.tolist() if prev_obs is not None else None,
                        prev_action,
                        decoder_input.tolist()
                    ])


                # saving reward and is_terminals
                self.agent.buffer.rewards.append(reward)
                self.agent.buffer.is_terminals.append(done)
                if prev_obs is not None and prev_action is not None:
                    prev_context = np.concatenate([prev_obs, np.eye(self.config['action_dim'])[prev_action]])
                    self.agent.buffer.prev_contexts.append(torch.FloatTensor(prev_context))
                else:
                    self.agent.buffer.prev_contexts.append(torch.zeros(self.config['input_dim'] + self.config['action_dim']))



                prev_obs = state
                prev_action = action
                state = next_state

                time_step +=1
                current_ep_reward += reward

                # update PPO agent
                if time_step % self.config['update_timestep'] == 0:
                    self.agent.update()

                # if continuous action space; then decay action std of ouput action distribution
                if self.config['has_continuous_action_space'] and time_step % self.config['action_std_decay_freq'] == 0:
                    self.agent.decay_action_std(self.config['action_std_decay_rate'], self.config['min_action_std'])

                # log in logging file
                if time_step % self.config['log_freq'] == 0:

                    # log average reward till last episode
                    log_avg_reward = log_running_reward / log_running_episodes
                    log_avg_reward = round(log_avg_reward, 4)

                    log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                    log_f.flush()

                    log_running_reward = 0
                    log_running_episodes = 0

                # printing average reward
                if time_step % self.config['print_freq'] == 0:

                    # print average reward till last episode
                    print_avg_reward = print_running_reward / print_running_episodes
                    print_avg_reward = round(print_avg_reward, 2)

                    logger.info("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                    print_running_reward = 0
                    print_running_episodes = 0

                # save model weights
                if time_step % self.config['save_model_freq'] == 0:
                    logger.info("--------------------------------------------------------------------------------------------")
                    logger.info("saving model at : " + self.directory)
                    self.agent.save_models(self.directory)
                    logger.info("model saved")
                    logger.info("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                    logger.info("--------------------------------------------------------------------------------------------")

                # break; if the episode is over
                if done:
                    break
                
            self.episode_rewards.append(current_ep_reward)
            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1

            i_episode += 1

        log_f.close()
        self.env.close()

        # print total training time
        logger.info("============================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        logger.info("Started training at (GMT) : ", start_time)
        logger.info("Finished training at (GMT) : ", end_time)
        logger.info("Total training time  : ", end_time - start_time)
        logger.info("============================================================================================")

        np.save(os.path.join(self.reward_folder, f"seq2seqBasic_{self.env_name}_step_rewards.npy"), np.array(self.step_rewards))
        np.save(os.path.join(self.reward_folder, f"seq2seqBasic_{self.env_name}_episode_rewards.npy"), np.array(self.episode_rewards))
        logger.info(f"Saved step_rewards and episode_rewards to {self.log_dir}")















class MindModelTrainer:
    def __init__(self, model: MindModel, config) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.horizon = self.config['horizon']
        self.model = model.to(self.device)

        self.MseLoss = nn.MSELoss()
        self.BCELoss = nn.BCELoss()

        self.model_dir = "MindModel_version"
        os.makedirs(self.model_dir, exist_ok=True)
        logger.info(f"Model directory created at {self.model_dir}")

        self.model_dir = os.path.join(self.model_dir, self.config['env_name'])
        os.makedirs(self.model_dir, exist_ok=True)
        logger.info(f"Specific Model directory created at {self.model_dir}")

        self.model_filename = f"mindmodel_{self.horizon}.pt"

    def train_(self, train_loader):
        self.model.train()
        best_loss = float('inf')
        total_loss = 0

        for batch in train_loader:
            obs = batch['obs'].to(self.device)               # [B, input_dim]
            action = batch['actions'].to(self.device)        # [B, horizon]
            next_obs = batch['next_obs'].to(self.device)     # [B, horizon, input_dim]
            reward = batch['rewards'].to(self.device)        # [B, horizon]
            done = batch['dones'].to(self.device)            # [B, horizon]

            batch_size = obs.size(0)
            obs = batch['obs'].to(self.device).unsqueeze(0)
            # Forward pass
            prev_obs = torch.randn(batch_size, self.config['input_dim']).to(self.device)
            output = self.model.horizon_predict(obs, action, prev_obs)  # horizon prediction

            pred_next_obs = output['next_obs']      # list of [B, input_dim]
            pred_reward = output['reward']          # list of [B, 1]
            pred_done = output['done']              # list of [B, 1]

            loss_next_obs = 0
            loss_reward = 0
            loss_done = 0

            for i in range(self.horizon):
                loss_next_obs += self.MseLoss(pred_next_obs[i], next_obs[:, i, :])
                loss_reward += self.MseLoss(pred_reward[i].squeeze(-1), reward[:, i])
                loss_done += self.BCELoss(pred_done[i].squeeze(-1), done[:, i])

            loss_next_obs /= self.horizon
            loss_reward /= self.horizon
            loss_done /= self.horizon

            total = loss_next_obs + loss_reward + loss_done

            self.model.optimizer.zero_grad()
            total.backward()
            self.model.optimizer.step()

            total_loss += total.item()
        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            self.model.save(self.model_dir, filename=self.model_filename)

        return avg_loss
    


    def train(self, train_loader, num_epochs=10, log_interval=1):
        best_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            total_loss = 0
            total_batches = 0

            total_done_correct = 0
            total_done_samples = 0

            for batch in train_loader:
                obs = batch['obs'].to(self.device)               # [B, input_dim]
                action = batch['actions'].to(self.device)        # [B, horizon]
                next_obs = batch['next_obs'].to(self.device)     # [B, horizon, input_dim]
                reward = batch['rewards'].to(self.device)        # [B, horizon]
                done = batch['dones'].to(self.device)            # [B, horizon]

                batch_size = obs.size(0)
                obs = obs.unsqueeze(0)  # [1, B, input_dim]

                prev_obs = torch.randn(batch_size, self.config['input_dim']).to(self.device)
                output = self.model.horizon_predict(obs, action, prev_obs)  # horizon prediction

                pred_next_obs = output['next_obs']  # list of [B, input_dim]
                pred_reward = output['reward']      # list of [B, 1]
                pred_done = output['done']          # list of [B, 1]

                loss_next_obs = 0
                loss_reward = 0
                loss_done = 0

                for i in range(self.horizon):
                    loss_next_obs += self.MseLoss(pred_next_obs[i], next_obs[:, i, :])
                    loss_reward += self.MseLoss(pred_reward[i].squeeze(-1), reward[:, i])
                    loss_done += self.BCELoss(pred_done[i].squeeze(-1), done[:, i])

                    # Binary accuracy logging
                    pred_labels = (pred_done[i].squeeze(-1) > 0.5).float()
                    total_done_correct += (pred_labels == done[:, i]).sum().item()
                    total_done_samples += done[:, i].numel()

                loss_next_obs /= self.horizon
                loss_reward /= self.horizon
                loss_done /= self.horizon

                total = loss_next_obs + loss_reward + loss_done

                self.model.optimizer.zero_grad()
                total.backward()
                self.model.optimizer.step()

                total_loss += total.item()
                total_batches += 1

            avg_loss = total_loss / total_batches
            avg_done_acc = total_done_correct / total_done_samples

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.model.save(self.model_dir, filename=self.model_filename)

            if epoch % log_interval == 0:
                logger.info(f"Epoch [{epoch}/{num_epochs}] | Loss: {avg_loss:.6f} | Done Accuracy: {avg_done_acc:.4f}")



    @torch.no_grad()
    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0
        total_done_correct = 0
        total_done_samples = 0

        for batch in test_loader:
            obs = batch['obs'].to(self.device)                 # [B, input_dim]
            action = batch['actions'].to(self.device)          # [B, horizon]
            next_obs = batch['next_obs'].to(self.device)       # [B, horizon, input_dim]
            reward = batch['rewards'].to(self.device)          # [B, horizon]
            done = batch['dones'].to(self.device)              # [B, horizon]

            batch_size = obs.size(0)
            obs = obs.unsqueeze(0)  # [1, B, input_dim]

            prev_obs = torch.randn(batch_size, self.config['input_dim']).to(self.device)
            output = self.model.horizon_predict(obs, action, prev_obs)

            pred_next_obs = output['next_obs']   # list of [B, input_dim]
            pred_reward = output['reward']       # list of [B, 1]
            pred_done = output['done']           # list of [B, 1]

            loss_next_obs = 0
            loss_reward = 0
            loss_done = 0

            for i in range(self.horizon):
                loss_next_obs += self.MseLoss(pred_next_obs[i], next_obs[:, i, :])
                loss_reward += self.MseLoss(pred_reward[i].squeeze(-1), reward[:, i])
                loss_done += self.BCELoss(pred_done[i].squeeze(-1), done[:, i])

                pred_labels = (pred_done[i].squeeze(-1) > 0.5).float()
                total_done_correct += (pred_labels == done[:, i]).sum().item()
                total_done_samples += done[:, i].numel()

            loss_next_obs /= self.horizon
            loss_reward /= self.horizon
            loss_done /= self.horizon
            total = loss_next_obs + loss_reward + loss_done
            total_loss += total.item()

        avg_loss = total_loss / len(test_loader)
        avg_done_acc = total_done_correct / total_done_samples

        logger.info(f"✅ Evaluation complete: Loss={avg_loss:.4f}, Done Accuracy={avg_done_acc:.4f}")
    
