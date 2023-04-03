import copy

import numpy as np
import torch

from configuration import *
from environment import Environment
from policy import Policy
from utils import Logger, plot

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

if __name__ == "__main__":
    logger = Logger(f"population_f{evaluations}_f{standard_deviation}_f{population_size}.log", 1000)

    env = Environment()
    policy = Policy(env.observation_space.shape[0], env.action_space.shape[0])
    policy.to(policy.device)

    perturbation = copy.deepcopy(policy)
    best_policy = copy.deepcopy(policy)

    for i in range(1, iterations + 1):
        best_score = -np.infty

        seed = np.random.randint(0, 2 ** 31 - 1)
        # Produce multiple perturbations
        for _ in range(population_size):
            # Create a perturbation
            for name, param in policy.named_parameters():
                perturbation.get_parameter(name).data = param + torch.randn_like(param) * standard_deviation

            # Evaluate the perturbation
            score = env.evaluate(perturbation, evaluations, seed=seed)

            # If this perturbation has the best score of the population at this iteration, keep it
            if score >= best_score:
                best_score = score
                best_policy = copy.deepcopy(perturbation)

        # Select the perturbation with the best score as the new policy
        policy = copy.deepcopy(best_policy)

        logger.log(i, best_score, policy)

    logger.close()
    plot(logger.filename)

    input()

    human_env = Environment(human=True)
    human_env.evaluate(policy, 100)
