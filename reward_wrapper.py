import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env, reward_weights):
        super().__init__(env)
        self.reward_weights = reward_weights
        self.most_recent_reward = None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Simulate the reward components
        info = info or {}
        if "forward_velocity" not in info and len(obs) > 2:
            try:
                info["forward_velocity"] = obs[3] if len(obs) > 3 else 0 # forward velocity assigned if given, else set to 0
                info["energy_penalty"] = -np.sum(np.abs(action))
                info["alive_bonus"] = 1.0 if not terminated else 0.0
            except:
                info["forward_velocity"] = 0.0
        
        shaped_reward = sum(
            self.reward_weights.get(k, 0.0) * info.get(k, 0.0) for k in self.reward_weights
        )

        self.most_recent_reward = reward
        return obs, reward, terminated, truncated, info
    
def run_episode_train(env_name, reward_weights, episodes=1, render_mode="rgb_array", model=None):
    env = gym.make(env_name, render_mode=render_mode)
    env = RewardShapingWrapper(env, reward_weights=reward_weights)
    
    frames = []
    #total_shaped = 0
    all_rewards_log = []
    total_reward = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            frames.append(env.render())
            done = terminated or truncated
            #total_shaped += shaped_reward
            total_reward += reward
            all_rewards_log.append(info)
    
    return total_reward,  all_rewards_log, frames

def train_agent(env_name, reward_weights, total_timesteps=100_000):
    def make_env():
        env = gym.make(env_name)
        return RewardShapingWrapper(env, reward_weights)
    
    env = make_vec_env(make_env, n_envs=1)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    return model






