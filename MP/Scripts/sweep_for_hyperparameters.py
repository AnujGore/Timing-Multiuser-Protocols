from objects.environment import GridTopologyEnv
from objects.agent import Agent
import numpy as np
import json
from train_model import train
from test_model import test
import argparse
import wandb

enviroment_parameters = json.load(open("MP/Scripts/config_files/simulation_config.json"))

n = enviroment_parameters["n"]
pgen = enviroment_parameters["pgen"]
pswap = enviroment_parameters["pswap"]
lifetime = enviroment_parameters["lifetime"]
init_state = False if enviroment_parameters["init_entangled"] == 0 else True

agent_parameters = json.load(open("MP/Scripts/config_files/model_config.json"))
num_episodes = agent_parameters["num_episodes"]
max_actions = agent_parameters["threshold_actions"]

all_actions = np.load("MP/Outputs/logs/4_nodes_actions.npy")
all_states =  np.load("MP/Outputs/logs/4_nodes_3_lifetime_states.npy")

def main():
    wandb.init(project='SP-swap-order', config=args)
    environment = GridTopologyEnv(n, pgen, pswap, all_actions, all_states, lifetime, init_state)
    agentQ = Agent(n, lifetime, all_actions, all_states)

    train(environment, agentQ, num_episodes, max_actions)
    mean_reward ,_ = test(environment, agentQ, num_episodes, max_actions)
    wandb.log({'mean_reward_test': mean_reward})

def argumentParser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exploration_rate', default=0.05, type=float, help='Probability of chossing random action')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='Learning Rate')
    parser.add_argument('--discount_factor', default=0.95, type=float, help='Discounting Factor')

    return parser

if __name__ == '__main__':
    global args
    args = argumentParser().parse_args()
    main()