from typing import Union

import torch
from unityagents import UnityEnvironment, BrainInfo


class ActionResult:
    def __init__(self,
                 state: Union[torch.FloatTensor, torch.cuda.FloatTensor],
                 reward: int,
                 done: bool) -> None:
        self.state = state
        self.reward = reward
        self.done = done

    @staticmethod
    def from_brain_info(brain_info: BrainInfo, device: str):
        return ActionResult(
            torch.from_numpy(brain_info.vector_observations[0]).float().to(device),
            brain_info.rewards[0],
            brain_info.local_done[0]
        )


class NavigationEnvironment:
    def __init__(self,
                 filename: str = './Banana.x86_64',
                 seed: int = 0,
                 device='cpu',
                 training_mode: bool = False) -> None:
        self.filename = filename
        self.seed = seed
        self.unit_env = UnityEnvironment(filename, seed=seed)
        self.brain_name = self.unit_env.brain_names[0]
        self.device = device
        self.training_mode = training_mode

    def action_size(self) -> int:
        return self.unit_env.brains[self.brain_name].vector_action_space_size

    def state_size(self) -> int:
        return self.unit_env.brains[self.brain_name].vector_observation_space_size

    def reset(self) -> ActionResult:
        return ActionResult.from_brain_info(self.unit_env.reset(train_mode=self.training_mode)[self.brain_name],
                                            self.device)

    def step(self, action: int) -> ActionResult:
        return ActionResult.from_brain_info(self.unit_env.step(action)[self.brain_name], self.device)
