import torch.cuda

from agent import Agent
from environment import NavigationEnvironment

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    env = NavigationEnvironment(device=DEVICE)
    agent = Agent(env.state_size(), env.action_size(), device=DEVICE)
    agent.load('./model_weights.pth')
    result = env.reset()
    action = agent.act(result.state)
    result = env.step(action)
    input("Press when to start")
    result = env.reset()
    score = 0
    while True:
        action = agent.act(result.state)
        result = env.step(action)
        score += result.reward

        if result.done:
            print(f'Total score: {score}')
            break
