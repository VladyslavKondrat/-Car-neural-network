import numpy as np
from agent.rl_agent import HybridAgent
from utils.logger import TensorboardLogger

class MockUnityEnv:
    """Pretends to be the C# Unity Game."""
    def __init__(self, num_beams=7):
        self.num_beams = num_beams
        self.step_count = 0
        self.max_steps_per_episode = 200

    def reset(self):
        self.step_count = 0
        # Return 15 random distances and a 0.0 (not crashed boolean)
        beams = np.random.uniform(0.1, 1.0, self.num_beams)
        return np.append(beams, 0.0)

    def step(self, throttle, steering, brake):
        self.step_count += 1
        
        # 1. Fake Physics: Generate random new sensor beams
        next_beams = np.random.uniform(0.1, 1.0, self.num_beams)
        
        # 2. Fake Logic: 5% chance the car crashes randomly
        is_crashed = np.random.rand() < 0.05
        done = is_crashed or (self.step_count >= self.max_steps_per_episode)
        
        # 3. Fake Rewards: +0.1 for surviving, -1.0 for crashing
        reward = -1.0 if is_crashed else 0.1
        
        # Combine into the exact array format your manager requested
        next_raw_state = np.append(next_beams, float(is_crashed))
        
        return next_raw_state, reward, done

def test_run():
    print("Starting Offline Mock Test...")
    
    num_beams = 15
    env = MockUnityEnv(num_beams=num_beams)
    
    # Import YOUR real, actual PyTorch brain
    agent = HybridAgent(input_size=num_beams)
    logger = TensorboardLogger(log_dir="runs/offline_mock_test")
    
    episodes = 500 # Just enough to test if the math breaks
    
    for episode in range(episodes):
        raw_state = env.reset()
        sensor_beams = raw_state[:-1]
        
        episode_reward = 0
        done = False
        
        while not done:
            # 1. Test your Actor network
            throttle, steering, brake_bool, log_prob, value = agent.select_action(sensor_beams)
            
            # 2. Test the mock environment
            next_raw_state, reward, done = env.step(throttle, steering, brake_bool)
            next_sensor_beams = next_raw_state[:-1]
            
            # 3. Test your Memory Buffer
            action_data = [throttle, steering, 1.0 if brake_bool else 0.0]
            agent.store_memory(sensor_beams, action_data, log_prob, reward, value, done)
            
            sensor_beams = next_sensor_beams
            episode_reward += reward

        # 4. TEST THE PPO MATH
        agent.train_step()
        
        print(f"Mock Episode {episode} | Reward: {episode_reward:.2f}")
        logger.log(episode, episode_reward, agent.get_losses())

if __name__ == '__main__':
    test_run()