Timing Multiuser Protocols
-------------------------

In this repository, we store and document all the resources necessary for quantifying the latency of different multiuser entanglement routing protocols. The purpose of this file is to note down the organisation and pinpoint the location of information necessary for understanding and reproducing our results

### Table of Contents
* About this project
* Structure of the repository
* Getting Started
* Usage
* License
* Contact

==========================

### About this project

The protocols we look to explore are the Shortest Path, MultiPath Greedy and MultiPath Cooperative or SP, MP-G and MP-C respectively. The first protocol was introduced by [Bugalho et al](https://arxiv.org/pdf/2103.14759), and in it, entanglement is generated on a predetermined path from a selected central node to each of the users. When generated, the fusion operation at the central node entangles the users giving us our needed state. The latter two protocols were introduced by [Sutcliffe et al](https://arxiv.org/pdf/2303.03334) and in these protocols, the need for a precomputed path is made unnecessary. In MP-G, entanglement is simulated and the protocol checks if there exists a valid path from the user to the central node, repeating entanglement generation if necessary. For MP-C, entanglement is simulated and the protocol checks if there exists a tree connecting the nodes, eliminating the need for the central node. The linked papers discuss the protocols in detail. Here we report the method used to quantify the latency of these protocols.

==========================

### Structure of the Repository

Below we document the structure of the repository:

```
root/
    ├── MP/
        ├── Docs/
            └── README.md
        ├── Scripts/
            └── utils.py
    ├── ShortestPath/
        ├── Docs/
            └── README.md
        ├── Scripts/
            ├── analytical_solutions.py
            ├── shortest_path.py
            └── utils.py      
        └── SP.ipynb

    ├── README.md       
    ├── LICENSE
    └── requirements.txt
```

==========================

### Getting started

We list the necessary libraries in the `requirements.txt` file and to recreate our results, running 

```
pip install -r requirements.txt
```

in the command line of the folder where the project needs to be stored.

==========================

### Usage

Within each folder, the `.ipynb` file exists which instruct how each result was reached and how it can be reproduced.

==========================

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

==========================

### Contact

Anuj Gore - anujgore@gmail.com