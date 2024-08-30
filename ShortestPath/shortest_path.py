import sys
sys.path.append('.')

import numpy as np
from helpers import good_factors


def monte_carlo(vertices, swap_samples, fusion_samples, p, q, k, ind = False):

    list_of_distances = []
    
    users = [(0, 0), (0, vertices-1), (vertices-1, 0), (vertices-1, vertices-1)]
    central_node = (int(vertices/2), int(vertices/2))

    for u in users:
        list_of_distances.append(int(np.abs(u[0] - central_node[0]) + np.abs(u[1] - central_node[1])))

    nesting_levels = []
    for d in list_of_distances:
        nesting_levels.append(int(np.log2(good_factors(d))[0]))

    calculator = MonteCarloFusion(max(nesting_levels), p, q, k)
    calculator.sample(swap_samples, fusion_samples)

    if (ind):
        return calculator.time_samples, calculator.tp_samples, calculator.tq_samples, calculator.tk_samples
    else:
        return calculator.tp_samples


class MonteCarloFusion():
    """Monte Carlo approach to calculating waiting time and fidelities. """

    def __init__(self, n, pgen, pswap, pfusion):
        self.params = {
            'pgen' : pgen,
            'pswap' : pswap,
            'pfusion': pfusion
        }
        self.n = n

    def sample(self, swap_samples, fusion_samples):
        self.time_samples, self.tp_samples, self.tq_samples, self.tk_samples = self.sample_fusion(swap_samples, fusion_samples)

    def sample_fusion(self, swap_samples, fusion_samples):
        success = []
        for _ in range(fusion_samples):
            fusion_success = False
            exp_val = 0
            while fusion_success != True:
                exp_val +=1
                fusion_success = np.random.random() <= self.params['pfusion']
            success.append(exp_val)

        
        return (np.ones(shape = (4, swap_samples))*(np.mean(success)))*self.sample_chain(swap_samples)
        

    def sample_chain(self, sample_size):
        this_time_samples = np.zeros(shape=(self.n+1, sample_size), dtype=int)
        this_tp_samples = np.zeros(shape=(self.n+1, sample_size), dtype=int)
        this_tq_samples = np.zeros(shape=(self.n+1, sample_size), dtype=int)
        for level in range(0, self.n+1):
            time_samps, tp_samps, tq_samps = self.sample_level(level, sample_size)
            this_time_samples[level] = time_samps
            this_tp_samples[level] = tp_samps
            this_tq_samples[level] = tq_samps

        return this_time_samples[-1], this_tp_samples[-1], this_tq_samples[-1], np.ones(shape = sample_size)

    def sample_level(self, n, sample_size):
        time_samples = np.zeros(sample_size)
        tp_samples = np.zeros(sample_size)
        tq_samples = np.zeros(sample_size)
        for k in range(sample_size):
            self.tq = 0
            tp = self.__sample_swap(n)
            time_samples[k] = tp + self.tq
            tp_samples[k] = tp
            tq_samples[k] = self.tq
        return time_samples, tp_samples, tq_samples

    def __sample_swap(self, n):
        if(n == 0):
            time = np.random.geometric(self.params['pgen'])
            return time
        else:
            tA = self.__sample_swap(n - 1)
            tB = self.__sample_swap(n - 1)

            tp = max(tA, tB)

            self.tq +=1

            swap_success = np.random.random() <= self.params['pswap']
            if(swap_success):
                return tp
            else:
                tp_retry = self.__sample_swap(n)
                return tp + tp_retry
