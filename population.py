import copy

import numpy as np
import torch

from environment import Environment
from policy import Policy

from configuration import *

from utils import Logger, plot

if __name__ == "__main__":
    logger = Logger(f"population_f{evaluations}_f{standard_deviation}_f{population_size}.log", 100)

    env = Environment()
    policy = Policy(env.observation_space.shape[0], env.action_space.shape[0])
    perturbation = copy.deepcopy(policy)

    for i in range(1, iterations + 1):
        best_score = -np.infty

        for _ in range(population_size):
            for name, param in policy.named_parameters():
                perturbation.get_parameter(name).data = param + torch.randn_like(param) * standard_deviation

            score = env.evaluate(perturbation, evaluations)

            if score >= best_score:
                best_score = score
                policy.load_state_dict(perturbation.state_dict())

        logger.log(i, best_score, policy)

    logger.close()
    plot(logger.filename)

