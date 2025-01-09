from objects.ppo import PPOPolicy, compute_advantages, determine_agent_loss_mb
from objects.environment import FixedPathEnv
import torch
import numpy as np
import time
import wandb
from utils.general_utils import check_if_tensor

#For wandb
class wandbArgs:
    wandb_project_name: str = "FixedPathPPO"
    wandb_dir: str = r"\rdata\ong\Anuj\University College London\Latency for MED protocols\Timing-Multiuser-Protocols\MP\SP"

#For environment
class environmentArgs:
    pgen = 0.7
    pswap = 0.5
    age_limit = 4
    n = 7

#For training
class hyperparametersArgs:
    hidden_dims = [64, 32, 16]
    gamma = 0.99
    lam = 0.95
    value_coefficient = 0.5
    entropy_coefficient = 0.01
    lr = 0.0003
    clip_eps = 0.2

#Fixed hyperparameters
class fixedHyperparameters:
    epochs = 10
    num_minibatches = 4
    save_freq = 100
    log_freq = 10
    total_timesteps = 500000
    num_steps = 1000
    num_steps = 128

    # Fill at runtime
    num_iterations = 0
    minibatch_size = 0

    wandb = wandbArgs
    environment = environmentArgs

if __name__ == "__main__":
    envArgs = environmentArgs()
    hyperArgs = hyperparametersArgs()
    fixedArgs = fixedHyperparameters()

    fixedArgs.batch_size = fixedArgs.num_steps
    fixedArgs.minibatch_size = int(fixedArgs.batch_size // fixedArgs.num_minibatches)
    fixedArgs.num_iterations = fixedArgs.total_timesteps // fixedArgs.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Add wandb configurations
    wandb.init(project = fixedArgs.wandb.wandb_project_name, dir = fixedArgs.wandb.wandb_dir)

    # Env setup
    myEnv = FixedPathEnv(n = envArgs.n, pgen = envArgs.pgen, pswap = envArgs.pswap, age_limit = envArgs.age_limit)

    # Policy setup
    myPolicy = PPOPolicy(n = envArgs.n, hidden_dims = hyperArgs.hidden_dims)

    # training configs
    optimizer = torch.optim.Adam(myPolicy.get_all_parameters(), lr = hyperArgs.lr)

    # Storage setup
    obs = torch.zeros(fixedArgs.num_steps, 4, envArgs.n**2, envArgs.n**2).to(device=device)
    actions = torch.zeros(fixedArgs.num_steps, envArgs.n**2, envArgs.n**2).to(device=device)
    rewards = torch.zeros(fixedArgs.num_steps).to(device=device)
    dones = torch.zeros(fixedArgs.num_steps).to(device=device)
    logprobs = torch.zeros(fixedArgs.num_steps).to(device=device)
    values = torch.zeros(fixedArgs.num_steps).to(device=device)

    #Initialize environment
    global_step = 0
    start_time = time.time()
    init_state, _ = myEnv.reset()
    next_done = torch.tensor(False, dtype=torch.float32).to(device=device)

    number_truncations = 0

    #Set initial state to current state
    current_state = init_state

    for iteration in range(1, fixedArgs.num_iterations+1):
        for step in range(0, fixedArgs.num_steps):
            global_step += 1
            #Collect trajectories
            with torch.no_grad():
                action = myPolicy.predict_action(current_state)
                value = myPolicy.predict_value(current_state)
                log_probs = myPolicy.log_prob(action)

            #Step through environment
            next_state, reward, terminated, truncated, _ = myEnv.act(action.numpy())
            next_done = torch.tensor(terminated or truncated, dtype=torch.float32).to(device=device)

            #Move to next state
            current_state = next_state

            rewards[step] = check_if_tensor(reward).to(device=device)
            actions[step] = check_if_tensor(action).to(device=device)
            values[step] = check_if_tensor(value).to(device=device)
            logprobs[step] = check_if_tensor(log_probs).to(device=device)
            obs[step] = check_if_tensor(current_state).to(device=device)
            dones[step] = check_if_tensor(next_done).to(device=device)


            #If the episode is terminated or truncated, reset the environment
            if terminated or truncated:
                if truncated: number_truncations += 1  
                current_state, _ = myEnv.reset()
                next_done = torch.tensor(False, dtype=torch.float32).to(device=device)
            
            #Find the GAE
            advantages, returns = compute_advantages(hyperArgs, rewards, values, dones)

            #Train actor-critic model
            batch_idxs = np.arange(fixedArgs.batch_size)
            for epoch in range(fixedArgs.epochs):
                np.random.shuffle(batch_idxs)
                for start in range(0, fixedArgs.batch_size, fixedArgs.minibatch_size):
                    end = start + fixedArgs.minibatch_size
                    minibatch_idxs = batch_idxs[start:end]

                    #Compute loss
                    loss = determine_agent_loss_mb(hyperArgs, minibatch_idxs, myPolicy, obs, logprobs, advantages, returns)

                    #Optimize the model
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            #Log the training progress
            if global_step % fixedArgs.log_freq == 0:
                wandb.log({"Reward": rewards[global_step - fixedArgs.log_freq:global_step].mean().item(), "Loss": loss.item(), "Number of Truncations": number_truncations})
                number_truncations = 0
                wandb.log({"Adjacency": wandb.Image(current_state[0])})