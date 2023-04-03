import gymnasium as gym
import torch.nn


class Environment:
    def __init__(self, human: bool = False, max_episode_steps: int = 500):
        """
        Create an environment wrapper for LunarLander-v2.
        :param human: True to render the environment
        :param max_episode_steps: maximum number of steps per episode
        """
        self.env = gym.make("LunarLander-v2", continuous=True, render_mode="human" if human else None)
        self.env = gym.wrappers.TimeLimit(self.env, max_episode_steps=max_episode_steps)

    def evaluate(self, policy: torch.nn.Module, N: int, seed: int = None):
        """
        Evaluate the policy for N episodes.
        :param policy: policy to evaluate
        :param N: number of evaluation episodes
        :param seed: seed for the evaluation
        :return: average reward
        """
        total_reward = 0
        for i in range(N):
            done = False
            truncated = False
            state = self.env.reset(seed=seed if seed is None else seed + i)[0]
            while not (done or truncated):
                action = policy(state)
                state, reward, done, truncated, info = self.env.step(action.cpu().detach().numpy())
                total_reward += reward
        return total_reward / N

    def __getattr__(self, name):
        return getattr(self.env, name)
