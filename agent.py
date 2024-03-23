import random
from typing import Union

import numpy as np
import torch

from memory import ReplayBuffer
from model import QNetwork


class Agent:
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 seed: int = 0,
                 device: str = 'cpu',
                 update_step: int = 8,
                 batch_size: int = 256,
                 buffer_size: int = 100_000,
                 gamma: float = 0.9,
                 tau: float = 1e-3,
                 lr: float = 5e-4,
                 training_mode: bool = False) -> None:
        self.action_size = action_size
        self.device = device

        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss()
        self.buffer = ReplayBuffer(buffer_size, device, state_size)

        self.t_step = 0
        self.update_step = update_step
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.training_mode = training_mode

        random.seed(seed)
        self.__soft_update()

    def act(self, state: Union[torch.FloatType, torch.cuda.FloatTensor], eps: float = 0.0) -> int:
        state = state.to(self.device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps or not self.training_mode:
            return torch.argmax(action_values).item()
        else:
            return random.choice(np.arange(self.action_size))

    def step(self,
             state: Union[torch.FloatTensor, torch.cuda.FloatTensor],
             action: float,
             reward: int,
             next_state: Union[torch.FloatTensor, torch.cuda.FloatTensor],
             done: bool) -> None:
        self.buffer.add(state, action, reward, next_state, done)

        self.t_step += 1
        if self.t_step % self.update_step == 0:
            self.__learn()

    def __learn(self) -> None:
        states, actions, rewards, next_states, dones = self.buffer.sample(min(self.action_size, len(self.buffer)))

        def _q_learning():
            next_Q_value = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            return rewards + (self.gamma * next_Q_value * (1 - dones.int()))

        def _double_q_learning():
            # https://arxiv.org/abs/1509.06461
            # https://davidrpugh.github.io/stochastic-expatriate-descent/pytorch/deep-reinforcement-learning/deep-q-networks/2020/04/11/double-dqn.html
            _actions = self.qnetwork_local(next_states).argmax(dim=1, keepdims=True)
            next_Q_value = self.qnetwork_target(next_states).gather(1, _actions)
            return rewards + (self.gamma * next_Q_value * (1 - dones.int()))

        Q_online_value = self.qnetwork_local(states).gather(1, actions)
        loss = self.loss_fn(Q_online_value, _double_q_learning())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.__soft_update()

    def __soft_update(self) -> None:
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def load(self, filename: str) -> None:
        self.qnetwork_local.load_state_dict(torch.load(filename))
