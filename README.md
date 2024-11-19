# Timing Multiuser Protocols

Here we develop analytical and monte-carlo based methods to time multi-user entanglement distribution protocols.

We develop Shortest Path (SP) protocol first, and can be run through the SP.ipynb file.  

We develop the timings in seconds as well as time-slots.

Within the `ShortestPath` folder, the analytical solutions and Monte-Carlo simulation results are stored. The `analytical_approaches.py` file contains the analytical while the `shortest_path.py` file has the Monte-Carlo simulation. The `misc` folder contains the additional functions which help in parallelization and miscellanous. All functionality is used in the `SP.ipynb` file where the results are reported. 

The `heralding models` folder contains the files which describe the method of heralding used by the protocol. Comparing these models gives us a better insight into the efficacy of each protocol.