from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime

import torch.nn

from environment import Environment
from policy import Policy


class Logger:
    def __init__(self, log: str, every: int, append: bool = False):
        """
        Create a class responsible for logging progress
        :param log: path to the logfile
        :param every: log every x iterations
        :param append: append to an existing logfile
        """
        Path.mkdir(Path("checkpoints"), exist_ok=True)
        self.filename = log
        self.log_file = open(Path("checkpoints") / log, "a" if append else "w")
        self.every = every
        self.start = datetime.now()

    def log(self, i: int, score: int, policy: torch.nn.Module):
        """
        Log an iteration of the algorithm
        :param i: iteration
        :param score: return of the evaluation
        :param policy: evaluated policy
        """
        if i % 10 == 0:
            print(f"{i}, {score} {datetime.now() - self.start}")
        print(f"{i}, {score}", file=self.log_file)
        if i % self.every == 0:
            policy.save(self.log_file.name + f".{i}.pt")

    def __getattr__(self, name):
        return getattr(self.log_file, name)


def plot(log):
    """
    Plot the return per episode
    :param log: log file where to retrieve the data
    """
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


# Code used to generate the final plots for the report
if __name__ == "__main__":
    # Plot zeroth-order method results
    zeroth_order_1 = np.loadtxt("checkpoints/FINAL/zeroth_order_f1_f0.05_f0.005.log", delimiter=",")
    zeroth_order_5 = np.loadtxt("checkpoints/FINAL/zeroth_order_f5_f0.05_f0.005.log", delimiter=",")

    plt.figure(figsize=(10, 6))
    factor = 200
    return_1 = np.convolve(zeroth_order_1[:, 1], np.ones(factor) / factor, mode="valid")
    return_5 = np.convolve(zeroth_order_5[:, 1], np.ones(factor) / factor, mode="valid")
    x = np.convolve(zeroth_order_1[:, 0], np.ones(factor) / factor, mode="valid")

    plt.plot(x * 1, return_1, label="$E = 1$")
    plt.plot(x * 5, return_5, label="$E = 5$")

    plt.legend()
    plt.title("Zeroth-order Optimisation")

    plt.xlabel("Episode")
    plt.ylabel("Return")

    plt.savefig("checkpoints/FINAL/zeroth_order.png", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Plot population-based results
    population_1_10 = np.loadtxt("checkpoints/FINAL/population_f1_f0.01_f10.log", delimiter=",")
    population_1_20 = np.loadtxt("checkpoints/FINAL/population_f1_f0.01_f20.log", delimiter=",")
    population_1_40 = np.loadtxt("checkpoints/FINAL/population_f1_f0.01_f40.log", delimiter=",")
    population_5_10 = np.loadtxt("checkpoints/FINAL/population_f5_f0.01_f10.log", delimiter=",")
    population_5_20 = np.loadtxt("checkpoints/FINAL/population_f5_f0.01_f20.log", delimiter=",")

    plt.figure(figsize=(10, 6))
    return_1_10 = np.convolve(population_1_10[:, 1], np.ones(factor) / factor, mode="valid")
    return_1_20 = np.convolve(population_1_20[:, 1], np.ones(factor) / factor, mode="valid")
    return_1_40 = np.convolve(population_1_40[:, 1], np.ones(factor) / factor, mode="valid")
    return_5_10 = np.convolve(population_5_10[:, 1], np.ones(factor) / factor, mode="valid")
    return_5_20 = np.convolve(population_5_20[:, 1], np.ones(factor) / factor, mode="valid")
    x = np.convolve(population_1_10[:, 0], np.ones(factor) / factor, mode="valid")

    plt.plot(x * 1 * 10, return_1_10, label="$E = 1, N = 10$")
    plt.plot(x * 1 * 20, return_1_20, label="$E = 1, N = 20$")
    plt.plot(x * 1 * 40, return_1_40, label="$E = 1, N = 40$")
    plt.plot(x * 5 * 10, return_5_10, label="$E = 5, N = 10$")
    plt.plot(x * 5 * 20, return_5_20, label="$E = 5, N = 20$")

    plt.legend()
    plt.title("Population-based Optimisation")

    plt.xlabel("Episode")
    plt.ylabel("Return")

    plt.savefig("checkpoints/FINAL/population.png", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()
