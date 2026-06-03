import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Import your neural networks from model.py
from agent.model import SelfDrivingBrain, ForwardDynamicsModel

class HybridAgent:
    def __init__(self, input_size, learning_rate=3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Initialize the Neural Networks
        self.brain = SelfDrivingBrain(input_size).to(self.device)
        # Forward model takes: current state + actions (throttle, steering, brake)
        self.forward_model = ForwardDynamicsModel(input_size + 3, input_size).to(self.device)
        
        # 2. Optimizers
        self.brain_optimizer = optim.Adam(self.brain.parameters(), lr=learning_rate)
        self.forward_optimizer = optim.Adam(self.forward_model.parameters(), lr=learning_rate)
        
        # 3. PPO Hyperparameters & Memory
        self.gamma = 0.99       # Discount factor for calculating future rewards
        self.eps_clip = 0.2     # PPO clipping parameter (epsilon)
        self.memory = []        
        
        # Logging metrics
        self.current_rl_loss = 0.0
        self.current_unsup_loss = 0.0

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # FIX: Call the probability function, NOT the raw forward pass
            cont_actions, brake_action, log_prob, value = self.brain.get_action_and_log_prob(state_tensor)
            
        # Extract the PyTorch tensors into standard Python numbers for Unity
        throttle = torch.clamp(cont_actions[0][0], -1.0, 1.0).item()
        steering = torch.clamp(cont_actions[0][1], -1.0, 1.0).item()
        brake = bool(brake_action.item())
        
        return throttle, steering, brake, log_prob, value

    def store_memory(self, state, action, log_prob, reward, value, done):
        """Saves the experience to the Replay Buffer."""
        self.memory.append((state, action, log_prob, reward, value, done))
        
        # Keep memory from growing infinitely and crashing your RAM
        if len(self.memory) > 100000:
            self.memory.pop(0)

    def train_step(self):
        #memory extract
        states = torch.FloatTensor(np.array([m[0] for m in self.memory])).to(self.device)
        old_log_probs = torch.stack([m[2] for m in self.memory]).squeeze().detach()
        rewards = [m[3] for m in self.memory]
        old_values = torch.stack([m[4] for m in self.memory]).squeeze().detach()
        dones = [m[5] for m in self.memory]

        # 1. Calculate Discounted Rewards (Returns)
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
            
        returns = torch.FloatTensor(returns).to(self.device)
        
        # 1.5 FIX: Only normalize if we have more than 1 frame of data
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # Ensure shapes match perfectly to remove the UserWarning
        returns = returns.squeeze()
        old_values = old_values.squeeze()

        # 2. Calculate Advantages (Actual Return - Critic's Guess)
        advantages = returns - old_values

        # 3. THE PPO UPDATE LOOP
        for _ in range(4): 
            # Recalculate probabilities with current network weights
            _, _, curr_log_probs, curr_values = self.brain.get_action_and_log_prob(states)
            curr_values = curr_values.squeeze()

            # The Ratio: New Prob / Old Prob
            ratios = torch.exp(curr_log_probs - old_log_probs)

            # Surrogate Losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Actor Loss: Negative minimum of the surrogates
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic Loss: Mean Squared Error between guess and actual return
            critic_loss = nn.MSELoss()(curr_values.squeeze(), returns)
            
            # Final PPO Loss
            rl_loss = actor_loss + 0.5 * critic_loss

            # Backpropagation
            self.brain_optimizer.zero_grad()
            rl_loss.backward()
            self.brain_optimizer.step()

        # Clear memory after PPO update (PPO is an On-Policy algorithm!)
        self.memory.clear()

    def get_losses(self):
        return {"RL_Loss": self.current_rl_loss, "Unsupervised_Loss": self.current_unsup_loss}

    def save_model(self, filepath):
        torch.save(self.brain.state_dict(), filepath)