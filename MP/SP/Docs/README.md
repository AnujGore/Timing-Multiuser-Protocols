# SP Documentation

This folder stores the software and resources for the SP routing. The difference of this folder from the ``SP_chain`` is the fact that the whole grid is used to evaluate the system rather than 4 chains. 

### Overview

Here’s a brief overview of the structure of the ``SP`` folder:

```
SP/
├── Scripts/
    ├── objects/
        ├── agent.py
        └── environment.py
    ├── config_files/
        ├── simulation_config.json      # JSON file for the simulation parameters
        └── model_config.json           # JSON file for the hyperparameters of the model
    ├── training.py
    └── utils.py                        # Utility functions for data processing
├── Outputs/
    ├── action_files/
        └── (Folder to store the actions)
├── Docs/                               
    └── README.md                       
```