# SP Documentation

This folder stores the software and resources for the SP routing. 

### Overview

Here’s a brief overview of the structure of the ``SP`` folder:

```
SP/
├── Scripts/
    ├── objects/
        ├── agent.py
        └── environment.py
    ├── config_files/
        ├── env_config.json             # JSON file for the environment parameters
        ├── simulation_config.json      # JSON file for the simulation parameters
        └── model_config.json           # JSON file for the hyperparameters of the model
    ├── training.py
    ├── plot_results.py                     
    └── utils.py                        # Utility functions for data processing
        ├── general_utils.py
        ├── model_utils.py
        └── SP_path_utils.py
├── Outputs/
    ├── model_paths/
        └── (Folder to store the models)
    ├── action_files/
        └── (Folder to store the actions)
├── Docs/                               
    └── README.md                       
```