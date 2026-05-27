import torch
import torch.nn as nn

class SelfDrivingBrain(nn.Module):
    """
    The main Actor network. 
    Input: Array of sensor beams.
    Outputs: Throttle [-1, 1], Steering [-1, 1], and Brake [True/False].
    """
    def __init__(self, num_beams):
        super(SelfDrivingBrain, self).__init__()
        
        # 1. The Shared Body
        # Extracts spatial features from the raw raycast distances.
        self.shared_layers = nn.Sequential(
            nn.Linear(num_beams, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 2. Head A: Continuous Actions (Throttle & Steering)
        # Tanh mathematically bounds the output exactly between -1.0 and 1.0.
        self.continuous_head = nn.Sequential(
            nn.Linear(64, 2),
            nn.Tanh()         
        )
        
        # 3. Head B: Discrete Action (Brake)
        # Sigmoid bounds the output between 0.0 and 1.0 (acting as a probability).
        self.discrete_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Pass input through the shared feature extractor
        features = self.shared_layers(x)
        
        # Split the features into the two separate heads
        cont_actions = self.continuous_head(features) # [throttle, steering]
        brake_prob = self.discrete_head(features)     # [brake_chance]
        
        # Convert the probability (0.0 to 1.0) into a strict boolean format (0 or 1)
        brake_bool = (brake_prob > 0.5).float() 
        
        return cont_actions, brake_bool


class ForwardDynamicsModel(nn.Module):
    """
    The Unsupervised (Self-Supervised) network.
    Input: Current sensor beams + The actions the car just took.
    Output: Prediction of what the next sensor beams will look like.
    """
    def __init__(self, input_size, output_size):
        # input_size = num_beams + 3 (throttle, steering, brake)
        # output_size = num_beams
        super(ForwardDynamicsModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        # x is the concatenated tensor of [current_beams, throttle, steering, brake]
        # It outputs a raw array of floats predicting the physics of the walls.
        return self.network(x)