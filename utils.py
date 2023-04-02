from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime

from environment import Environment
from policy import Policy


class Logger:
	def __init__(self, log, every, append=False):
		Path.mkdir(Path("checkpoints"), exist_ok=True)
		self.filename = log
		self.log_file = open(Path("checkpoints") / log, "a" if append else "w")
		self.every = every
		self.start = datetime.now()

	def log(self, i, score, policy):
		if i % 10 == 0:
			print(f"{i}, {score} {datetime.now() - self.start}")
		print(f"{i}, {score}", file=self.log_file)
		if i % self.every == 0:
			policy.save(self.log_file.name + f".{i}.pt")

	def __getattr__(self, name):
		return getattr(self.log_file, name)


def plot(log):
	data = np.loadtxt(Path("checkpoints") / log, delimiter=",")
	plt.figure(figsize=(10, 6))
	factor = data.shape[0] // 100
	x_smoothed = np.convolve(data[:, 0], np.ones(factor) / factor, mode="valid")
	y_smoothed = np.convolve(data[:, 1], np.ones(factor) / factor, mode="valid")
	plt.plot(x_smoothed, y_smoothed, label=f"Return (averaged over {factor} episodes)")
	plt.legend()
	plt.xlabel("Episode")
	plt.ylabel("Return")
	plt.savefig(Path("checkpoints") / (log + ".png"), dpi=300, bbox_inches="tight", pad_inches=0)
	plt.show()


if __name__ == "__main__":
	env = Environment(human=True)
	plot("zeroth_order_f1_f0.001_f0.001.log")
	policy = Policy(env.observation_space.shape[0], env.action_space.shape[0])
	policy.load("checkpoints/zeroth_order_f1_f0.001_f0.001.log.217000.pt")

	score = env.evaluate(policy, 10, seed=0)
	print(score)
