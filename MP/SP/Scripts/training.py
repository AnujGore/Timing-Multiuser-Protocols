import numpy as np
from utils.general_utils import generate_random_action

all_actions = []

action, all_actions = generate_random_action(3, all_actions, users = np.array([(0, 0), (0, 2), (2, 0), (2, 2)]), cn_loc = (1, 1))