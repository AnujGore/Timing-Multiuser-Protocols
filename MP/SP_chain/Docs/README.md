# SP chain

This folder stores the software and resources for determining the swap order for a linear chain of repeaters. The purpose of this documentation is to ensure clarity on the process, format, and usage of the SP routing protocol. We can include the fusion probability at the end to emulate MP protocols

### Overview

Here’s a brief overview of the structure of the ``SwapOrderRL`` folder:

```
MP/
├── Scripts/
    ├── objects/
        ├── agent.py
        └── environment.py
    ├── config_files/
        ├── simulation_config.json      # JSON file for the simulation parameters
        └── model_config.json           # JSON file for the hyperparameters of the model
    ├── test_model.py
    ├── train_model.py
    └── utils.py                        # Utility functions for data processing
├── Outputs/
    ├── plot_results.py
    ├── logs/
        └── (Folder to store the states and actions)
├── Docs/                               
    └── README.md                       
```

### Key files

1. ``environment.py``: In this file, we create the environment the protocols routing works by. We define the step function which transforms the state depending on the action provided and all the necessary functionality for a reinforcement learning environment.

2. ``agent.py``: In this file, we create the agent which has the epsilon greedy algorithm and updates the q-table via the Q-algorithm presented in the [linked paper](https://arxiv.org/pdf/2303.00777). 

3. ``train_model.py``: In this file, we train the reinforcement learning agent.

4. ``test_model.py``: In this file, we test the reinforcement learning agent.