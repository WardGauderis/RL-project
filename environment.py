import gymnasium as gym


class Environment:
	def __init__(self, human=False):
		self.env = gym.make("LunarLander-v2", continuous=True, render_mode="human" if human else None)
		self.env = gym.wrappers.TimeLimit(self.env, max_episode_steps=500)

	def evaluate(self, policy, N, seed=None):
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
