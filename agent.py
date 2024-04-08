from dqn import DeepQNetwork
import torch
import torch.optim as optim
import torch.nn as nn
import random
import math
import numpy as np
import pickle
from replay_memory import ReplayMemory, Transition

MAX_MEMORY = 100_000
MAX_SEED = 10_000
LR = 1e-3
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.01
EPS_DECAY = 300
LOG_EVERY_EPISODE = 10
EPISODES = 1000
SAVE_EVERY_EPISODE = 10
CHECKPOINT_FILE = 'checkpoint.pth'
BEST_MODEL_FILE = 'best-model.pth'

class TetrisAgent():
    def __init__(self, n_observations, n_actions, env_step_func, env_reset_func, env_next_steps_func, env_score_func, plot_rewards_func):
        # if GPU is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # set class properties
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.env_step_func = env_step_func
        self.env_reset_func = env_reset_func
        self.env_next_steps_func = env_next_steps_func
        self.env_score_func = env_score_func
        self.plot_rewards_func = plot_rewards_func
        self.policy_net = DeepQNetwork(n_observations, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(MAX_MEMORY)
        self.steps_done = 0
        self.predict_map = None

    def select_action(self, state, eps_threshold=0.1):
        self.steps_done += 1
        inputs = len(state)
        sample = random.random()
        if sample > eps_threshold:
            self.policy_net.eval()
            with torch.no_grad():
                self.predict_map = self.policy_net(state)[:, 0]
                final_action =  torch.argmax(self.predict_map)
            self.policy_net.train()
        else:
            final_action = torch.tensor([[random.randrange(inputs)]], device=self.device, dtype=torch.long)

        return final_action

    def optimize_model(self):
        # Sample minibatch with size N from memory
        if len(self.memory) < (MAX_SEED+BATCH_SIZE):
            return

        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        state_batch = torch.stack(tuple(state for state in batch.state))
        reward_batch = torch.from_numpy(np.array(batch.reward, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in batch.next_state))

        q_values = self.policy_net(state_batch)
        self.policy_net.eval()
        with torch.no_grad():
            next_prediction_batch = self.policy_net(next_state_batch)
        self.policy_net.train()

        y_batch = torch.cat(
            tuple(reward if done else reward + GAMMA * prediction for reward, done, prediction in
                    zip(reward_batch, batch.done, next_prediction_batch)))[:, None]

        self.optimizer.zero_grad()
        criterion = nn.MSELoss()
        loss = criterion(q_values, y_batch)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _get_tensor_stack(self, objects):
        tensors = tuple(torch.tensor(t, dtype=torch.float32, device=self.device) for t in objects)
        return torch.stack(tensors)

    def play_game(self, eps_threshold=1):
        self.steps_done = 0
        ep_rewards = []
        done = False
        state = torch.tensor(self.env_reset_func(), dtype=torch.float32, device=self.device)
        while not done:
            # get next actions and states from game
            next_actions, next_states = zip(*self.env_next_steps_func().items())
            next_states_tensors = self._get_tensor_stack(next_states)
            max_action = self.select_action(next_states_tensors, eps_threshold).item()
            next_state = next_states_tensors[max_action, :]
            action = next_actions[max_action]

            # step through game
            _, reward, done, truncated = self.env_step_func(list(action))

            # set reward tensor
            ep_rewards.append(reward)
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)

            # Store the transition in memory
            self.memory.push(state, reward, next_state, done)

            # Move to the next state and append episode rewards
            state = next_state

            # Perform one step of the optimization (on the policy network)
            self.optimize_model()

            # set done if truncated episode
            done = truncated if truncated else done

        return ep_rewards

    def train(self):
        state = None
        num_episode = 0
        episode_totals = []
        reward_record = 0
        max_score = 0
        min_score = 0
        avg_steps = []
        avg_rewards = []

        # initialize memory seed
        while len(self.memory) < MAX_SEED:
            self.play_game()

        # play episodes for training
        for num_episode in range(1, EPISODES):
            # Initialize the environment and get its state
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * num_episode / EPS_DECAY)

            # play game and return rewards
            ep_rewards = self.play_game(eps_threshold)

            # append rewards and plot graph
            episode_totals.append(sum(ep_rewards))
            self.plot_rewards_func(show_result=False, totals=episode_totals)

            avg_steps.append(self.steps_done)
            avg_rewards.append(sum(ep_rewards))
            if sum(ep_rewards) > reward_record:
                reward_record = sum(ep_rewards)
                with open('model/' + BEST_MODEL_FILE, 'wb') as f:
                    pickle.dump(self.policy_net, f)

            if self.env_score_func() > max_score:
                max_score = self.env_score_func()

            min_score = self.env_score_func() if min_score == 0 or self.env_score_func() < min_score else min_score

            if num_episode % SAVE_EVERY_EPISODE == 0:
                with open('model/' + CHECKPOINT_FILE, 'wb') as f:
                    pickle.dump(self.policy_net, f)

            if num_episode % LOG_EVERY_EPISODE == 0:
                print(f'[{num_episode}] mem: {len(self.memory)} avg steps: {sum(avg_steps)/LOG_EVERY_EPISODE:.2f} eps:  {eps_threshold:.2f} min score:  {min_score} max score:  {max_score} avg reward: {sum(avg_rewards)/LOG_EVERY_EPISODE:.2f} top reward: {reward_record:.2f} ')

                # reset variables
                avg_steps = []
                avg_rewards = []
                max_score = 0

        self.plot_rewards_func(show_result=True, totals=episode_totals)

