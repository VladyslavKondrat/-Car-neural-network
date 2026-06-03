import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

from agent.rl_agent import HybridAgent 
from utils.logger import TensorboardLogger

def main():
    print("Starting PPO Brain Server...")
    
    env = UnityEnvironment(file_name=None)
    env.reset()
    behavior_name = list(env.behavior_specs.keys())[0]
    
    num_beams = 7
    agent = HybridAgent(input_size=num_beams)
    logger = TensorboardLogger()
    
    episodes = 10000
    
    for episode in range(episodes):
        env.reset()
        episode_reward = 0
        done = False
        
        raw_state = get_unity_state(env, behavior_name) 
        sensor_beams = raw_state[:-1]  
        is_crashed = bool(raw_state[-1]) 
        
        while not done:
            # --- CHANGE 1: UNPACK LOG PROBABILITIES AND VALUES ---
            throttle, steering, brake_bool, log_prob, value = agent.select_action(sensor_beams)
            
            # Format actions for Unity C#
            continuous_actions = np.array([[throttle, steering]], dtype=np.float32)
            discrete_actions = np.array([[1 if brake_bool else 0]], dtype=np.int32)
            unity_actions = ActionTuple(continuous=continuous_actions, discrete=discrete_actions)
            
            # Send to C# physics engine
            env.set_actions(behavior_name, unity_actions)
            env.step()
            
            # Gather feedback from Unity
            next_raw_state, reward, done_from_unity = get_unity_feedback(env, behavior_name)
            next_sensor_beams = next_raw_state[:-1]
            is_crashed = bool(next_raw_state[-1])
            
            if is_crashed:
                reward -= 1.0 
                done = True   
            elif done_from_unity:
                done = True
                
            # --- CHANGE 2: NEW STORAGE FORMAT ---
            # We must pass the log_prob and value into memory.
            # Note: We group the actions into a single array/tuple to match the loss math.
            action_data = [throttle, steering, 1.0 if brake_bool else 0.0]
            agent.store_memory(sensor_beams, action_data, log_prob, reward, value, done)
            
            # (Note: agent.train_step() is REMOVED from here. No per-frame training!)
            
            sensor_beams = next_sensor_beams
            episode_reward += reward

        # --- CHANGE 3: TRAIN AT THE END OF THE EPISODE ---
        # Now that the car has finished its run or crashed, we pass the entire
        # trajectory data to PPO to calculate rewards and optimize weights.
        print(f"Episode {episode} finished. Updating weights...")
        agent.train_step() 
        
        # Log and save progress
        print(f"Episode {episode} | Total Reward: {episode_reward}")
        logger.log(episode, episode_reward, agent.get_losses())
        
        if episode % 100 == 0:
            agent.save_model(f"saved_models/ppo_car_brain_ep{episode}.pth")

    env.close()

# --- UNITY HELPER FUNCTIONS ---
def get_unity_state(env, behavior_name):
    decision_steps, _ = env.get_steps(behavior_name)
    return decision_steps.obs[0][0] 

def get_unity_feedback(env, behavior_name):
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    if len(terminal_steps) > 0:
        return terminal_steps.obs[0][0], terminal_steps.reward[0], True
    return decision_steps.obs[0][0], decision_steps.reward[0], False

if __name__ == '__main__':
    main()