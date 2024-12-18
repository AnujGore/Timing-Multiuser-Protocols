import numpy as np
from utils.general_utils import generate_random_action

n = 3; 
users = np.array([(0, 0), (0, n-1), (n-1, 0), (n-1, n-1)]); cn_loc = (int(np.floor(n/2)), int(np.floor(n/2)))
all_actions = []

action, all_actions = generate_random_action(n, all_actions, users, cn_loc)
