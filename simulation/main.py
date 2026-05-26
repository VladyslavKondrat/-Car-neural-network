import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

from agent.rl_agent import
from utils.logger import


def main():
    print("Starting...")

    env = UnityEnvironment(file_name=None)
    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]
    #bridge - name of the "brain" to send and receive data

    num_beams = 7    #may be changed
    agent = the_agent(input_size=num_beams)
    logger = TensorBoardLogger()



    #THE loop
    episodes = 10000
    for episode in range(episodes):
        env.reset()
        episode_reward=0
        done=False

        raw_state = get_unity_state(env, behavior_name) 
        sensor_beams = raw_state[:-1]  # Slice: Everything except the last item
        is_crashed = bool(raw_state[-1]) # Slice: The boolean flag  


        while not done:
            
            # --- THINK (PyTorch) ---
            # Pass the beams to your network to get the actions
            # throttle: [-1, 1], steering: [-1, 1], brake: [True/False]
            throttle, steering, brake_bool = agent.select_action(sensor_beams)
            
            # --- FORMAT FOR C# (ML-Agents) ---
            # Unity expects continuous actions as a float array, discrete as an int array
            continuous_actions = np.array([[throttle, steering]], dtype=np.float32)
            discrete_actions = np.array([[1 if brake_bool else 0]], dtype=np.int32)
            
            unity_actions = ActionTuple(continuous=continuous_actions, discrete=discrete_actions)
            
            # --- ACT ---
            # Send the actions to C# and advance the physics by one frame
            env.set_actions(behavior_name, unity_actions)
            env.step()
            
            # --- OBSERVE RESULT ---
            # Get the new data from C# after the move
            next_raw_state, reward, done_from_unity = get_unity_feedback(env, behavior_name)
            
            next_sensor_beams = next_raw_state[:-1]
            is_crashed = bool(next_raw_state[-1])
            
            # Manager's Rule: 'crashed' tells us if the car failed.
            if is_crashed:
                reward -= 1.0 # Optional: Add a heavy penalty for crashing
                done = True   # Force the episode to end
            elif done_from_unity:
                done = True
                
            # --- LEARN (RL + Unsupervised) ---
            # Save this experience to memory.
            # The agent will use this to update the Actor-Critic (rewards) 
            # AND the Forward Model (predicting next_sensor_beams).
            agent.store_memory(sensor_beams, (throttle, steering, brake_bool), reward, next_sensor_beams, done)
            agent.train_step() 
            
            # --- PREPARE FOR NEXT FRAME ---
            sensor_beams = next_sensor_beams
            episode_reward += reward

        # --- END OF EPISODE ---
        print(f"Episode {episode} | Total Reward: {episode_reward}")
        logger.log(episode, episode_reward, agent.get_losses())
        
        # Save your progress periodically
        if episode % 100 == 0:
            agent.save_model(f"saved_models/car_brain_ep{episode}.pth")

    # Cleanup
    env.close()

# --- HELPER FUNCTIONS ---
# (Abstracted Unity boilerplate to keep the main loop clean)

def get_unity_state(env, behavior_name):
    """Extracts the raw array from Unity's DecisionSteps."""
    decision_steps, _ = env.get_steps(behavior_name)
    # Returns the first agent's observation array
    return decision_steps.obs[0][0] 

def get_unity_feedback(env, behavior_name):
    """Extracts the next state, reward, and done flag."""
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    
    if len(terminal_steps) > 0:
        # The car crashed or ran out of time
        next_state = terminal_steps.obs[0][0]
        reward = terminal_steps.reward[0]
        return next_state, reward, True
    else:
        # The car is still driving
        next_state = decision_steps.obs[0][0]
        reward = decision_steps.reward[0]
        return next_state, reward, False

if __name__ == '__main__':
    main()






