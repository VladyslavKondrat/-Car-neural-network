import os
from torch.utils.tensorboard import SummaryWriter

class TensorboardLogger:
    """
    Handles writing training metrics to TensorBoard.
    Run 'tensorboard --logdir runs' in your terminal to view the dashboard.
    """
    def __init__(self, log_dir="runs/self_driving_experiment_1"):
        # Create the directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Initialize the PyTorch SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"Logger initialized. Tracking data in: {log_dir}")

    def log(self, episode, total_reward, losses):
        """
        Writes the current episode's data to the TensorBoard graphs.
        
        Args:
            episode (int): The current training loop iteration.
            total_reward (float): The final score the car got this episode.
            losses (dict): Dictionary containing the RL and Unsupervised losses.
        """
        # 1. Log the Reward (This is the most important graph!)
        # We want to see this line go UP over time.
        self.writer.add_scalar("Performance/Total_Reward", total_reward, episode)
        
        # 2. Log the Losses
        if losses:
            # Unsupervised Loss (Forward Model): We want this to go DOWN 
            # as it gets better at predicting the physics of the walls.
            if "Unsupervised_Loss" in losses:
                self.writer.add_scalar("Loss/Unsupervised_MSE", losses["Unsupervised_Loss"], episode)
            
            # RL Loss: This graph will look messy and fluctuate, which is normal for PPO,
            # but it tells you if the network's weight updates are exploding.
            if "RL_Loss" in losses:
                self.writer.add_scalar("Loss/RL_Policy", losses["RL_Loss"], episode)

    def close(self):
        """Closes the writer cleanly when training is done."""
        self.writer.close()