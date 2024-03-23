import random
from typing import Union, Tuple

import torch


class ReplayBuffer:

    def __init__(self,
                 buffer_size: int,
                 device: str,
                 size: int,
                 seed: int = 0) -> None:
        self.buffer_size = buffer_size
        self.next_value = 0
        self.choices = set()
        random.seed(seed)

        self.state_memory = torch.zeros(buffer_size, size).float().to(device)
        self.next_state_memory = torch.zeros(buffer_size, size).float().to(device)
        self.actions_memory = torch.zeros(buffer_size).long().to(device)
        self.rewards_memory = torch.zeros(buffer_size).float().to(device)
        self.dones_memory = torch.zeros(buffer_size).bool().to(device)

    def add(self,
            state: Union[torch.FloatTensor, torch.cuda.FloatTensor],
            action: float,
            reward: int,
            next_state: Union[torch.FloatTensor, torch.cuda.FloatTensor],
            done: bool) -> None:
        self.state_memory[self.next_value, :] = state
        self.next_state_memory[self.next_value, :] = next_state
        self.actions_memory[self.next_value] = torch.tensor(action).int().to(self.actions_memory.device)
        self.rewards_memory[self.next_value] = torch.tensor(reward).int().to(self.actions_memory.device)
        self.dones_memory[self.next_value] = torch.tensor(done).bool().to(self.actions_memory.device)

        self.choices.add(self.next_value)
        self.next_value = (self.next_value + 1) % self.buffer_size

    def sample(self, batch_size: int) -> Tuple[Union[torch.FloatTensor, torch.cuda.FloatTensor],
                                               Union[torch.LongTensor, torch.cuda.LongTensor],
                                               Union[torch.FloatTensor, torch.cuda.FloatTensor],
                                               Union[torch.FloatTensor, torch.cuda.FloatTensor],
                                               Union[torch.BoolTensor, torch.cuda.BoolTensor]]:
        selected_indices = random.sample(self.choices, batch_size)

        return (self.state_memory[selected_indices, :],
                self.actions_memory[selected_indices].view(len(selected_indices), 1),
                self.rewards_memory[selected_indices].view(len(selected_indices), 1),
                self.next_state_memory[selected_indices, :],
                self.dones_memory[selected_indices].view(len(selected_indices), 1).int())

    def __len__(self) -> int:
        return len(self.choices)
