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
        # We use Adam to update the weights of both models
        self.brain_optimizer = optim.Adam(self.brain.parameters(), lr=learning_rate)
        self.forward_optimizer = optim.Adam(self.forward_model.parameters(), lr=learning_rate)
        
        # 3. Memory Buffer
        self.memory = []
        self.batch_size = 64
        
        # Logging metrics
        self.current_rl_loss = 0.0
        self.current_unsup_loss = 0.0

    def select_action(self, state, explore=True):
        """Passes the sensor beams through the Brain to get a driving move."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad(): # We don't calculate gradients when just driving
            cont_actions, brake_bool = self.brain(state_tensor)
            
        throttle = cont_actions[0][0].item()
        steering = cont_actions[0][1].item()
        brake = brake_bool[0][0].item() > 0.5 # Convert to actual python boolean
        
        # --- EXPLORATION LOGIC ---
        # If 'explore' is true, you would add some random math (noise) to the 
        # throttle and steering here so the car tries new things.
        
        return throttle, steering, brake

    def store_memory(self, state, action, reward, next_state, done):
        """Saves the experience to the Replay Buffer."""
        self.memory.append((state, action, reward, next_state, done))
        
        # Keep memory from growing infinitely and crashing your RAM
        if len(self.memory) > 100000:
            self.memory.pop(0)

    def train_step(self):
        """The core learning algorithm combining RL and Unsupervised Learning."""
        # Don't train if we haven't collected enough data yet
        if len(self.memory) < self.batch_size:
            return

        # 1. Sample a random batch of memories
        # (In reality, you'd use a proper sampling function here)
        batch = self.memory[-self.batch_size:] 
        
        states = torch.FloatTensor([m[0] for m in batch]).to(self.device)
        actions = torch.FloatTensor([m[1] for m in batch]).to(self.device)
        rewards = torch.FloatTensor([m[2] for m in batch]).to(self.device)
        next_states = torch.FloatTensor([m[3] for m in batch]).to(self.device)
        
        # --- UNSUPERVISED LEARNING (The Forward Model) ---
        # Goal: Predict the next_states based on current states + actions
        state_action_cat = torch.cat((states, actions), dim=1)
        predicted_next_states = self.forward_model(state_action_cat)
        
        # Unsupervised Loss: How wrong was the prediction? (Mean Squared Error)
        unsup_loss = nn.MSELoss()(predicted_next_states, next_states)
        
        # Update Forward Model
        self.forward_optimizer.zero_grad()
        unsup_loss.backward()
        self.forward_optimizer.step()

        # --- REINFORCEMENT LEARNING (The Brain) ---
        # Goal: Maximize points. 
        # (Note: This is a highly abstracted placeholder. Real RL algorithms like 
        # PPO or SAC require calculating Advantages, Log Probs, and Value targets here).
        
        # rl_loss = calculate_ppo_loss(...)
        rl_loss = torch.tensor(0.0, requires_grad=True).to(self.device) # Placeholder
        
        # Update Brain
        self.brain_optimizer.zero_grad()
        rl_loss.backward()
        self.brain_optimizer.step()
        
        # Save metrics for logging
        self.current_unsup_loss = unsup_loss.item()
        self.current_rl_loss = rl_loss.item()

    def get_losses(self):
        return {"RL_Loss": self.current_rl_loss, "Unsupervised_Loss": self.current_unsup_loss}

    def save_model(self, filepath):
        torch.save(self.brain.state_dict(), filepath)