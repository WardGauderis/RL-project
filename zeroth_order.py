import copy

import torch

from configuration import *
from environment import Environment
from policy import Policy
from utils import Logger, plot

seed = 0
torch.manual_seed(seed)

if __name__ == "__main__":
	logger = Logger(f"zeroth_order_f{evaluations}_f{standard_deviation}_f{learning_rate}.log", 1000)

	env = Environment()
	policy = Policy(env.observation_space.shape[0], env.action_space.shape[0])
	policy.to(policy.device)

	positive = copy.deepcopy(policy)
	negative = copy.deepcopy(policy)

	for i in range(1, iterations + 1):
		ps = {}
		for name, param in policy.named_parameters():
			perturbation = torch.randn_like(param) * standard_deviation
			ps[name] = perturbation
			positive.get_parameter(name).data = param + perturbation
			negative.get_parameter(name).data = param - perturbation

		pos = env.evaluate(positive, evaluations)
		neg = env.evaluate(negative, evaluations)

		for name, param in policy.named_parameters():
			gradient = (pos - neg) / 2 * ps[name]
			param.data += learning_rate * gradient

		logger.log(i, max(pos, neg), policy)

	logger.close()
	plot(logger.filename)

	input()

	human_env = Environment(human=True)
	human_env.evaluate(policy, 100)
