import copy

import numpy as np
import torch
import torch.nn as nn
import torch.cuda
from collections import deque
import random


class Mario:
    def __init__(self, state, action, save_directory):
        self.state = state
        self.action = action
        self.save_directory = save_directory
        self.memory = deque(maxlen=1000000)
        self.batch_size = 32

        self.use_cuda = torch.cuda.is_available()
        self.net = GameNet(self.state, self.action).float()
        if self.use_cuda:
            self.net = self.net.to(device='cuda')
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.exploration_rate = 1.0
        self.exploration_decay = 0.999998
        self.exploration_min = 0.1
        self.current_step = 0
        self.save_every = 5e5
        self.gamma = 0.9
        self.burnin = 1e4
        self.learn_every = 3
        self.sync_every = 1e4

    def act(self, state):
        """ Epsilon greedy action determination, selects the action and updates the step """

        # Exploration State
        if np.random.rand() < self.exploration_rate:
            action_index = np.random.randint(self.action)

        # Exploitation State
        else:
            state = state.__array__()
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model="online")
            action_index = torch.argmax(action_values, axis=1).item()

            self.exploration_rate *= self.exploration_decay
            self.exploration_rate = max(self.exploration_rate, self.exploration_min)
        self.current_step += 1
        return action_index

    def cache(self, state, next_state, action, reward, done):
        """ Store this play experience in memory """
        state = state.__array__()
        next_state = next_state.__array__()

        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        """ Recall a random batch from memory """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def estimate(self, state, action):
        current_Q = self.net(state, model='online')[
            np.arange(0, self.batch_size), action
        ]
        return current_Q

    def update_Q_online(self, estimate, target):
        loss=self.loss_fn(estimate, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (
            self.save_directory / f"mario_game_network_{int(self.current_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"GameNet has been saved to {save_path} at step # {self.current_step}")

    def learn(self):
        if self.current_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.current_step % self.save_every == 0:
            self.save()

        if self.current_step < self.burnin:
            return None, None

        if self.current_step % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.recall()

        est = self.estimate(state, action)
        tgt = self.target(reward, next_state, done)

        loss = self.update_Q_online(est, tgt)

        return est.mean().item(), loss

    @torch.no_grad()
    def target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model='target')[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1-done.float())*self.gamma * next_Q).float()



class GameNet(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super().__init__()
        c, h, w = input_dimension

        if h!=84:
            raise ValueError(f"Expecting input height of 84, got: {h}.")
        if w!=84:
            raise ValueError(f"Expecting input width of 84, got: {w}.")

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dimension),
        )

        self.target = copy.deepcopy(self.online)

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == 'online':
            return self.online(input)
        elif model == 'target':
            return self.target(input)

