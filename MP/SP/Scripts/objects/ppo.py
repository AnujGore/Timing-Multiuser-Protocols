from agent import Agent, DualWeightsNet

class PPO:
    def __init__(self, n, hidden_dims):
        self.actor = Agent(n, hidden_dims = hidden_dims)
        self.critic = DualWeightsNet(n, hidden_dims, 1)

    def learn(self, total_timesteps):
        t_so_far = 0
        while t_so_far < total_timesteps:
            t_so_far += 1
            # Do something
            pass