# MultiPath (MP) Documentation

This folder stores the software and resources for the MP multiparty routing protocol. The purpose of this documentation is to ensure clarity on the process, format, and usage of the two MultiPath routing protocols - MP-G and MP-C.

### Overview

Here’s a brief overview of the structure of the ``MP`` folder:

```
MP/
├── Scripts/               
    ├── environment.py 
    ├── train_model.py
│   └── utils.py        # Utility functions for data processing
│
├── Docs/               # Documentation folder (this folder)
│   └── README.md       # Overview of data generation
```

### Key files

1. ``environment.py``: In this file, we create the environment the protocols routing works by. We define the step function which transforms the state depending on the action provided and all the necessary functionality for a reinforcement learning environment.

2. ``train_model.py``: In this file, we train the reinforcement learning agent.