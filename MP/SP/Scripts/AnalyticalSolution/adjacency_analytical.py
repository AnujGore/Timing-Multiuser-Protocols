import torch
from rich.progress import track

def correct_adj(this_adj):
    this_adj += torch.rot90(torch.fliplr(this_adj))
    return this_adj

def generate_entanglement(this_system_adj, pgen, protocol):
    this_adj = this_system_adj[0]
    this_ent_adj = this_system_adj[1]
    elementary_links = torch.diagonal(this_adj, 1)

    links_to_form_loc = torch.where(elementary_links == 0)[0] 
    elementary_links = torch.bernoulli(elementary_links[links_to_form_loc], pgen)

    rng = torch.arange(1, len(this_adj))
    all_rng = torch.arange(len(this_adj))

    this_adj[links_to_form_loc, links_to_form_loc+1] = elementary_links

    if protocol == "SenderReceiver":
        this_ent_adj[rng-1, rng-1] += 1
    elif protocol == "MeetInTheMiddle":
        this_ent_adj[all_rng, all_rng] += 0.5

    this_adj = correct_adj(this_adj)
    this_system_adj = torch.stack((this_adj, this_ent_adj, this_system_adj[2]))
    return this_system_adj

def perform_swap(this_system_adj, this_repeater_loc, pswap):
    this_adj = this_system_adj[0]
    this_repeater_adj = this_system_adj[2]

    can_swap = torch.bernoulli(torch.zeros(len(this_repeater_loc)), pswap).to(torch.bool)

    for i, node in enumerate(can_swap):
        possible_nodes_to_connect = torch.nonzero(this_adj[this_repeater_loc[i]]).T[0]
        if len(possible_nodes_to_connect) == 2:
            if node:
                this_adj[possible_nodes_to_connect[0], possible_nodes_to_connect[1]] = 1
                this_adj[possible_nodes_to_connect[1], possible_nodes_to_connect[0]] = 1
            
            this_adj[this_repeater_loc[i], possible_nodes_to_connect[0]] = 0
            this_adj[possible_nodes_to_connect[0], this_repeater_loc[i]] = 0
            this_adj[this_repeater_loc[i], possible_nodes_to_connect[1]] = 0
            this_adj[possible_nodes_to_connect[1], this_repeater_loc[i]] = 0

            this_repeater_adj[this_repeater_loc[i], this_repeater_loc[i]] +=1

    this_system_adj = torch.stack((this_adj, this_system_adj[1], this_repeater_adj))
    return this_system_adj


# def MonteCarloSP(n, pgen, pswap, protocol, iter = 10000):

#     ent_attempts_iter = []
#     swap_attempts_iter = []
    
#     for _ in track(range(iter), description="Simulating..."):
#         mySystemAdj = torch.zeros(size = (3, n, n))
#         repeater_loc = torch.arange(1, n-1)
#         while mySystemAdj[0][0][-1] == 0:
#             mySystemAdj = generate_entanglement(mySystemAdj, pgen, protocol)
#             mySystemAdj = perform_swap(mySystemAdj, repeater_loc, pswap)

#         ent_attempts = torch.diagonal(mySystemAdj[1]).max()
#         swap_attempts = torch.diagonal(mySystemAdj[2]).max()

#         ent_attempts_iter.append(ent_attempts)
#         swap_attempts_iter.append(swap_attempts)

#     return (torch.tensor(ent_attempts_iter).mean(), torch.tensor(swap_attempts_iter).mean(), torch.tensor(ent_attempts_iter).std(), torch.tensor(swap_attempts_iter).std())

import torch
import torch.multiprocessing as mp

def simulate_worker(rank, n, pgen, pswap, protocol, iter, return_dict):
    """
    Worker function to perform Monte Carlo simulations in parallel.
    """
    ent_attempts_iter = []
    swap_attempts_iter = []

    for _ in range(iter):
        mySystemAdj = torch.zeros(size=(3, n, n))
        repeater_loc = torch.arange(1, n-1)
        while mySystemAdj[0][0][-1] == 0:
            mySystemAdj = generate_entanglement(mySystemAdj, pgen, protocol)
            mySystemAdj = perform_swap(mySystemAdj, repeater_loc, pswap)

        ent_attempts = torch.diagonal(mySystemAdj[1]).max()
        swap_attempts = torch.diagonal(mySystemAdj[2]).max()

        ent_attempts_iter.append(ent_attempts.item())
        swap_attempts_iter.append(swap_attempts.item())

    return_dict[rank] = (ent_attempts_iter, swap_attempts_iter)


def MonteCarloSP_torch_parallel(n, pgen, pswap, protocol, iter=10000, num_workers=None):
    """
    Monte Carlo simulation with parallelized for-loop using torch.multiprocessing.
    """
    if num_workers is None:
        num_workers = mp.cpu_count()

    # Split iterations among workers
    iterations_per_worker = iter // num_workers
    extra_iterations = iter % num_workers
    worker_iterations = [iterations_per_worker + (1 if i < extra_iterations else 0) for i in range(num_workers)]

    # Shared dictionary to collect results
    manager = mp.Manager()
    return_dict = manager.dict()

    # Create and start processes
    processes = []
    for rank, iters in enumerate(worker_iterations):
        p = mp.Process(target=simulate_worker, args=(rank, n, pgen, pswap, protocol, iters, return_dict))
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Aggregate results
    ent_attempts_iter = []
    swap_attempts_iter = []
    for rank in range(num_workers):
        ent_attempts, swap_attempts = return_dict[rank]
        ent_attempts_iter.extend(ent_attempts)
        swap_attempts_iter.extend(swap_attempts)

    # Convert to tensors for statistics
    ent_attempts_tensor = torch.tensor(ent_attempts_iter)
    swap_attempts_tensor = torch.tensor(swap_attempts_iter)

    return (
        ent_attempts_tensor.mean(),
        swap_attempts_tensor.mean(),
        ent_attempts_tensor.std(),
        swap_attempts_tensor.std(),
    )
