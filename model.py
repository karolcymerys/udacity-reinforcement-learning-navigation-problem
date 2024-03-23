from typing import Union

import torch
import torch.nn as nn


class QNetwork(nn.Module):

    def __init__(self, state_size: int, action_size: int, seed: int = 0, hidden_size_factor: int = 5) -> None:
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(in_features=state_size, out_features=hidden_size_factor*state_size)
        self.fc2 = nn.Linear(in_features=hidden_size_factor*state_size, out_features=action_size)
        self.relu = nn.ReLU()

    def forward(self, state: Union[torch.FloatTensor, torch.cuda.FloatTensor]
                ) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]:
        x = self.relu(self.fc1(state))
        return self.fc2(x)
