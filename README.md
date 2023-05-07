# Fundamental Project
## Gradient-Free Policy Optimisation

---
Ward Gauderis - 0588485  
Reinforcement Learning - Vrije Universiteit Brussel  
01/06/2023

---

### Overview of the project files

- `configuration.py`: Contains the hyperparameters of the agents
- `population.py`: Contains the main loop of the population-based agent
- `zeroth_order.py`: Contains the main loop of the zeroth-order agent
- `environment.py`: Contains the environment which evaluates the agents
- `policy.py`: Contains the parametric policy
- `utils.py`: Contains utils for logging and plotting

### How to run the agents

```bash
python population.py
python zeroth_order.py
```

These commands will run the agents in the environment with the configured hyperparameters. The logs as well as checkpoints and graphs of the evaluation
returns will be saved in the `checkpoints` folder.

The final logs, checkpoints and graphs used in the report are already present in the `checkpoints/FINAL` folder. The final graphs can be replotted by running

```bash
python utils.py
```