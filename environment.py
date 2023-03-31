import gymnasium as gym
import torch


class Environment:
    def __init__(self):
        self.env = gym.make("LunarLander-v2", continuous=True)

    def evaluate(self, policy, N):
        total_reward = 0
        for _ in range(N):
            done = False
            state = self.env.reset()
            while not done:
                action = policy(torch.tensor(state[0], dtype=torch.float32))
                observation, reward, done, truncated, info = self.env.step(action.detach().numpy())
                total_reward += reward
        return total_reward / N

    def __getattr__(self, name):
        return getattr(self.env, name)
