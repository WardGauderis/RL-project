import copy

import torch

from configuration import *
from environment import Environment
from policy import Policy
from utils import Logger, plot

if __name__ == "__main__":
    logger = Logger(f"zeroth_order_f{evaluations}_f{standard_deviation}_f{learning_rate}.log", 100)

    env = Environment()
    policy = Policy(env.observation_space.shape[0], env.action_space.shape[0])

    positive = copy.deepcopy(policy)
    negative = copy.deepcopy(policy)

    for i in range(1, iterations + 1):
        for name, param in policy.named_parameters():
            perturbation = torch.randn_like(param) * standard_deviation
            positive.get_parameter(name).data = param + perturbation
            negative.get_parameter(name).data = param - perturbation

        pos = env.evaluate(positive, evaluations)
        neg = env.evaluate(negative, evaluations)

        for name, param in policy.named_parameters():
            gradient = (pos - neg) / 2 * positive.get_parameter(name).data
            param.data += learning_rate * gradient

        logger.log(i, max(pos, neg), policy)

    logger.close()
    plot(logger.filename)
