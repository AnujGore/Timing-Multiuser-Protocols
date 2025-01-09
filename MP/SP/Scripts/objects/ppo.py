from objects.agent import Agent, DualWeightsNet
import torch
from utils.general_utils import flip_extend_list

class PPOPolicy:
    def __init__(self, n, hidden_dims):
        self.actor = Agent(n, hidden_dims = flip_extend_list(hidden_dims))
        self.critic = DualWeightsNet(n**2, hidden_dims, 1)

    def get_all_parameters(self):
        all_parameters = self.actor.this_parameters + list(self.critic.parameters())
        return all_parameters

    def predict_action(self, state):
        return self.actor.predict_action(state)
    
    def log_prob(self, action):
        return self.actor.log_prob(action)
    
    def entropy(self):
        return self.actor.entropy()
    
    def predict_value(self, state):
        if type(state) != 'torch.Tensor':
            state = torch.Tensor(state)

        if state.ndim == 3:
            state = state[1, :, :] #Training on age matrix only
        elif state.ndim == 4:
            state = state[:, 1, :, :] #Training on age matrix only

        return self.critic(state)


def determine_agent_loss_mb(args, minibatch_idxs, agent, states, oldlogprob, advantages, returns):
    huber_loss = torch.nn.HuberLoss()

    # Policy Objective Function
    newlogprobs = agent.log_prob(states[minibatch_idxs])
    ratios = torch.exp(newlogprobs - oldlogprob[minibatch_idxs])
    surr1 = ratios * advantages[minibatch_idxs]
    surr2 = torch.clamp(ratios, 1 - args.clip_eps, 1 + args.clip_eps) * advantages[minibatch_idxs]
    actor_loss = -torch.min(surr1, surr2).mean()
    
    # Value Function Loss
    values = agent.predict_value(states[minibatch_idxs]).reshape(-1)    
    value_loss = huber_loss(values, returns[minibatch_idxs])

    # Entropy Loss
    entropy_loss = agent.entropy().mean()

    # Total Loss
    loss = actor_loss + args.value_coefficient * value_loss - args.entropy_coefficient * entropy_loss

    return loss

def compute_advantages(args, rewards, values, dones):
    advantages = torch.zeros_like(rewards)
    gae = 0 # Generalized Advantage Estimation
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            delta = rewards[t] - values[t]
            gae = delta
        else:
            delta = rewards[t] + (1 - dones[t]) * args.gamma * values[t + 1] - values[t]
            gae = delta + (1 - dones[t]) * args.gamma * args.lam * gae
            advantages[t] = gae
    
    returns = advantages + values
    return advantages, returns