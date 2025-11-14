#!/usr/bin/python3
import gymnasium as gym
import ale_py
gym.envs.registration.register_envs(ale_py)
import numpy as np
from collections import deque
from agent import DQN_Agent
from assignment3_utils import FrameStacker, ReplayMemory
import datetime
import tensorflow as tf


def train(env_name='ALE/Pong-v5', episodes=200, batch_size=8,
          update_rate=10):
    # Hyperparameters
    gamma = 0.95     # Discount factor
    
    # Create Pong environment
    env = gym.make(env_name, render_mode="rgb_array")
    state_size = (84, 80, 4)
    action_size = env.action_space.n
    agent = DQN_Agent(state_size, action_size, gamma)

    # GPU info
    print("Num GPUs:", len(tf.config.list_physical_devices('GPU')))
    print("Using:", tf.config.list_physical_devices('GPU'))

    # Enable GPU memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("GPU memory growth enabled.")
        except RuntimeError as e:
            print(e)

    load_model_name = ""
    if load_model_name != "":
        agent.load(load_model_name)

    episodes = episodes
    batch_size = batch_size
    skip_start = 90  # Skip first frames
    total_time = 0
    all_rewards = 0
    input_frames = 4
    done = False
    frame_stack = FrameStacker(num_frames=4)
    memory_size = 10000
    memory = ReplayMemory(memory_size)
    loss = 0

    # TensorBoard setup
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(
        f"runs/batch{batch_size}_rate{update_rate}_{current_time}"
    )

    # Track last 5 rewards
    last_5_rewards = deque(maxlen=5)

    scores = []
    avg_rewards = []

    for e in range(episodes):
        total_reward = 0
        game_score = 0
        sum_10_games = 0
        state_reset = env.reset()
        state = frame_stack.reset(state_reset[0])
        stack = frame_stack.get_current_stack()  # Ensure 4 frames ready
    

        for skip in range(skip_start):  # Skip game start
            env.step(0)
        
        for time in range(2500):
            total_time += 1
            action = agent.act(state)
            # Take step
            next_state, reward, done, _, _ = env.step(action)
            
            # Update stacked frames
            next_state_processed = frame_stack.update(next_state)

            # Save experience
            memory.add(state, action, reward, next_state_processed, done)

            state = next_state_processed
            game_score += reward
            total_reward += reward
            if done:
                scores.append(game_score)
                all_rewards += game_score
                sum_10_games += game_score
                avg_10_games = sum_10_games / (e % 10 + 1)

                # Reward logs
                last_5_rewards.append(total_reward)
                avg_last_5 = np.mean(last_5_rewards)

                # TensorBoard logs
                with file_writer.as_default():
                    tf.summary.scalar('Episode Reward', total_reward, step=e)
                    tf.summary.scalar('Avg Reward (last 5 episodes)', avg_last_5, step=e)
                
                print("Ep {}/{} | Score: {} | Avg10: {:.2f} | TotalR: {:.2f} | AvgR: {:.2f} | Time: {} | TotalT: {}"
                      .format(e+1, episodes, game_score, avg_10_games,
                              total_reward, all_rewards/(e+1), time, total_time))
                
                break

            # Train every 25 steps after warm-up
            if len(memory.states) > 2000 and total_time % 25 == 0:
                loss = agent.replay(memory, batch_size)
                with file_writer.as_default():
                    tf.summary.scalar('Loss', loss, step=e)
                
        # Update target network
        if total_time % agent.update_rate == 0:
            agent.update_target_model()

        # Save model periodically
        if e % agent.update_rate == 0:
            fname = f'models/10k-memory_{e}-games.weights.h5'
            print(f'Saving: {fname}')
            agent.save(fname)
            sum_10_games = 0

        # Early stop if agent wins
        if game_score > 15.0:
            break

    agent.save('models/5k-memory_100-games.weights.h5')

    return scores, avg_rewards

if __name__ == "__main__":
    train()
