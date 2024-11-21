# Shortest Path Documentation

This folder stores the software and resources for the Shortest Path multiparty routing protocol. The purpose of this documentation is to ensure clarity on the process, format, and usage of the Shortest Path routing method.

### Overview

Here’s a brief overview of the structure of the ``ShortestPath`` folder:

```
ShortestPath/
├── Scripts/            
│   ├── analytical_solutions.py
│   ├── shortest_path.py    
│   └── utils.py        # Utility functions for data processing
│
├── Docs/               # Documentation folder (this folder)
│   └── README.md       # Overview of data generation
|
└── SP.ipynb
```

### Key files

1. ``analytical_solutions.py``: In this file, the analytical solutions for latency in terms of imperfect entanglement generation and imperfect repeater swapping is given. These have been published by respective authors.

2. ``shortest_path.py``: In this file, a class method which stores the network can be simulated. Depending on a variable (set to `False`), the individual time spent in entangling, swapping and fusion can be returned. Currently employing the doubling scheme and requires ``int(log2(repeaters))`` repeaters.

3. ``SP.ipynb``: In this file, the above two files are imported and the results are compared. The analysis of the SP protocol is also documented in this file.