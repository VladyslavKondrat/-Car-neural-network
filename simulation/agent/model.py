import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical

class SelfDrivingBrain(nn.Module):
    def __init__(self, num_beams):
        super(SelfDrivingBrain, self).__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(num_beams, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # ACTOR HEADS
        self.continuous_head = nn.Sequential(
            nn.Linear(64, 2),
            nn.Tanh()         
        )
        self.discrete_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # CRITIC HEAD (New for PPO)
        self.value_head = nn.Linear(64, 1)
        
        # STANDARD DEVIATION (New for PPO)
        self.continuous_std = nn.Parameter(torch.zeros(2)) 

    def forward(self, x):
        features = self.shared_layers(x)
        
        # Actor
        action_means = self.continuous_head(features)
        brake_prob = self.discrete_head(features)
        
        # Critic
        state_value = self.value_head(features)
        
        # Now it properly returns 3 things!
        return action_means, brake_prob, state_value

    def get_action_and_log_prob(self, state_tensor):
        """Creates probability distributions and samples actions from them."""
        # This will now successfully unpack 3 values
        action_means, brake_prob, state_value = self.forward(state_tensor)
        
        # 1. Continuous Distribution (Throttle, Steering)
        action_std = torch.exp(self.continuous_std) 
        continuous_dist = Normal(action_means, action_std)
        cont_actions = continuous_dist.sample()
        cont_log_probs = continuous_dist.log_prob(cont_actions).sum(dim=-1)
        
        # 2. Discrete Distribution (Brake)
        discrete_dist = Categorical(probs=torch.cat([1-brake_prob, brake_prob], dim=-1))
        brake_action = discrete_dist.sample() 
        brake_log_prob = discrete_dist.log_prob(brake_action)
        
        # Total log probability
        total_log_prob = cont_log_probs + brake_log_prob
        
        return cont_actions, brake_action, total_log_prob, state_value


class ForwardDynamicsModel(nn.Module):
    """The Unsupervised (Self-Supervised) network."""
    def __init__(self, input_size, output_size):
        super(ForwardDynamicsModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.network(x)