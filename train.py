from collections import deque
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from agent import Agent
from environment import NavigationEnvironment

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

N_EPISODES = 5_000

EPS_START = 1.0
EPS_MIN = 0.01
EPS_DECAY = 0.99

GAMMA = 0.9
TAU = 1e-3
LR = 5e-4


def plot_score(values: List[int]) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(values)), values)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


if __name__ == '__main__':
    env = NavigationEnvironment(device=DEVICE, training_mode=True)
    agent = Agent(env.state_size(), env.action_size(), device=DEVICE, training_mode=True, gamma=GAMMA, tau=TAU, lr=LR)
    eps = EPS_START

    scores = []
    scores_window = deque(maxlen=100)
    with tqdm(range(1, N_EPISODES + 1)) as episodes:
        for i_episode in episodes:

            prev_result = env.reset()
            score = 0
            for t in range(1, 301):
                action = agent.act(prev_result.state, eps)
                result = env.step(action)
                agent.step(prev_result.state, action, result.reward, result.state, result.done)
                score += result.reward

                if result.done:
                    break

                prev_result = result

            eps = max(EPS_MIN, EPS_DECAY * eps)
            scores.append(score)
            scores_window.append(score)
            episodes.set_postfix({'Avg reward': np.mean(scores_window)})

            if np.average(scores[-100:]) >= 13.0:
                break

    print(f'Environment resolved in {i_episode} episodes.')
    torch.save(agent.qnetwork_local.state_dict(), f'model_weights.pth')

    plot_score(scores)
