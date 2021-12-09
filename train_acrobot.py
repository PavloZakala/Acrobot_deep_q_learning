import random

import gym
import numpy as np
import torch
from torch import nn

from tools import ReplayMemory

torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAMMA = 0.99  # gamma parameter for the long term reward
MEMORY = 10000  # Replay memory capacity
LR = 1e-3
BATCH_SIZE = 256

NET_SAVE_STEP = 100
NET_UPDATE_SPET = 10  # Number of episodes to wait before updating the target network
MIN_SAMPLES_FOR_TRAINING = 1000

INIT_EXP_DELAY_VALUE = 5
NUM_EPISODES = 1000

# Exponential decay
exp_decay = np.exp(-np.log(
    INIT_EXP_DELAY_VALUE) / NUM_EPISODES * 6)  # We compute the exponential decay in such a way the shape of the exploration profile does not depend on the number of iterations
exploration_profile = [INIT_EXP_DELAY_VALUE * (exp_decay ** i) for i in range(NUM_EPISODES)]


class DQN(nn.Module):

    def __init__(self, state_space_dim, action_space_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state_space_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64 * 2),
            nn.ReLU(),
            nn.Linear(64 * 2, action_space_dim)
        )

    def forward(self, x):
        x = x.to(device)
        return self.model(x)


class Agent:

    def __init__(self, env):
        self.env = env
        # Init models
        self.replay_mem = ReplayMemory(MEMORY)

        state_space_dim = env.observation_space.shape[0]
        action_space_dim = env.action_space.n

        # Initialize models
        self.main_net = DQN(state_space_dim, action_space_dim).to(device)
        self.target_net = DQN(state_space_dim, action_space_dim).to(device)
        self.target_net.load_state_dict(self.main_net.state_dict())

        # Initialize the optimizer
        self.optimizer = torch.optim.Adam(self.main_net.parameters(), lr=LR)

        # Initialize the loss function
        self.loss_fn = nn.SmoothL1Loss()

    def get_best_action(self, state):

        with torch.no_grad():
            self.main_net.eval()
            state = torch.tensor(state, dtype=torch.float32)  # Convert the state to tensor
            out = self.main_net(state)

        best_action = int(out.argmax())
        return best_action, out.cpu().numpy()

    def get_action(self, state, eps):

        if eps == 0:
            return self.get_best_action(state)

        eps = max(eps, 1e-8)

        with torch.no_grad():
            self.main_net.eval()
            state = torch.tensor(state, dtype=torch.float32)
            out = self.main_net(state)

        actions_prob = nn.functional.softmax(out / eps, dim=0).cpu().numpy()

        possible_actions = np.arange(0, actions_prob.shape[-1])
        action = np.random.choice(possible_actions, p=actions_prob)

        return action, out.cpu().numpy()

    def update_step(self):
        global GAMMA, BATCH_SIZE

        batch = self.replay_mem.sample(BATCH_SIZE)

        # Create tensors
        states = torch.tensor(np.array([s[0] for s in batch]), dtype=torch.float32, device=device)
        next_states = torch.tensor(np.array([s[2] for s in batch if s[2] is not None]), dtype=torch.float32,
                                   device=device)
        next_states_mask = torch.tensor(np.array([s[2] is not None for s in batch]), dtype=torch.bool)
        actions = torch.tensor(np.array([s[1] for s in batch]), dtype=torch.int64, device=device)
        rewards = torch.tensor(np.array([s[3] for s in batch]), dtype=torch.float32, device=device)


        self.main_net.train()
        q_values = self.main_net(states)

        state_action_values = q_values.gather(1, actions.unsqueeze(1).cuda())

        with torch.no_grad():
            self.target_net.eval()
            q_values_target = self.target_net(next_states)
        next_state_max_q_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_max_q_values[next_states_mask] = q_values_target.max(dim=1)[0].detach()


        expected_q_values = (rewards + (next_state_max_q_values * GAMMA))

        # Loss
        loss = self.loss_fn(state_action_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(self.main_net.parameters(), 2) # For training stability
        self.optimizer.step()

    def step(self, state, eps):
        # Choose the action
        action, q_values = self.get_action(state, eps)

        # Apply the action and get the next state, reward and done
        next_state, reward, done, _ = self.env.step(action)

        if done:
            next_state = None

        # Update replay memory
        self.replay_mem.push(state, action, next_state, reward)

        # training only if we have enough samples in the replay memory
        if len(self.replay_mem) > MIN_SAMPLES_FOR_TRAINING:
            self.update_step()

        return next_state, reward, done

    def update_target_network(self):
        print('Updating target network...')
        self.target_net.load_state_dict(self.main_net.state_dict())

    def save_current_state(self, path):
        torch.save(self.main_net.state_dict(), path)

    def load_current_state(self, path):
        self.main_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(torch.load(path))


if __name__ == '__main__':

    torch.manual_seed(0)
    np.random.seed(0)

    env = gym.make('Acrobot-v1')
    env.seed(0)  # Set a random seed for the environment (reproducible results)

    agent = Agent(env)

    for episode, eps in enumerate(exploration_profile):

        state = env.reset()
        score = 0
        done = False

        while not done:
            next_state, reward, done = agent.step(state, eps)

            score += reward
            state = next_state

            # env.render()

        # Update the target network every target_net_update_steps episodes
        if episode % NET_UPDATE_SPET == 0:
            agent.update_target_network()

        if episode % NET_SAVE_STEP == 0:
            agent.save_current_state(f"model/model_{episode}.pth")
        print(f"Episode: {episode + 1}, Score: {score}")

    env.close()

    agent.save_current_state("model/model.pth")
