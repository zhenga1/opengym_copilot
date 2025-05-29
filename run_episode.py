import gymnasium as gym
import argparse
## reward weights:
"""
info = {
    "forward_velocity": 0.9,
    "energy_penalty": 0.2,
    "alive_bonus": 1.0
}
"""

def run_episode(env_name, reward_weights, episodes, render_mode="human"):
    env = gym.make(env_name, render_mode=render_mode)
    obs, _ = env.reset()
    total_shaped_reward_all_eps = 0
    total_reward_all_eps = 0
    all_rewards_log = []
    frames = []

    done = False
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_shaped_reward = 0
        ep_reward = 0
        rewards_log = []

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # shaped reward ? reward weights ? 
            # reward_weights = knobs turned by the user (me)
            # info - per step reward components from the environment
            # shaped_reward = total reward per step
            # fake info based on rewards
            if not info:
                 print("Info not provided, values are MANUALLY SHAPED.")
                 info = {
                    "forward_velocity": obs[3], # hull angle speed
                    "energy_penalty": -abs(action).sum(),
                    "alive_bonus": 1.0 if not terminated else 0.0
                }
            shaped_reward = sum(w * info.get(k, 0.0) for k, w in reward_weights.items())
            rewards_log.append((reward, shaped_reward, info))
            ep_shaped_reward += shaped_reward
            ep_reward += reward

            frame = env.render()
            frames.append(frame)

        print(f"Episode {ep+1}: Total Shaped Reward = {ep_shaped_reward:.2f}")
        total_shaped_reward_all_eps += ep_shaped_reward
        total_reward_all_eps += ep_reward
        all_rewards_log.append(rewards_log)

    env.close()
    return total_shaped_reward_all_eps, total_reward_all_eps, all_rewards_log, frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Program to Run Episode",
        description="Run an episode of an environment with a set of reward weighht parameters"
    )
    parser.add_argument("--episodes", type=int, default=1000) # default 1000 episodes
    parser.add_argument("--env_name", type=str, default="BipedalWalker-v3") # environment name
    #parser.add_argument("--reward_parameters", type=list, default=None) # avoid list parameters for now

    ## get some reward parameters
    parser.add_argument("--weight_forward_velocity", type=float, default=0.9)
    parser.add_argument("--weight_energy_penalty", type=float, default=0.2)
    parser.add_argument("--weight_alive_bonus", type=float, default=0.1) # ??? 

    args = parser.parse_args()
    episodes, env_name, forward_velocity, energy_penalty, alive_bonus = args.episodes, args.env_name, args.weight_forward_velocity, args.weight_energy_penalty, args.weight_alive_bonus

    reward_weights = {
        "forward_velocity": forward_velocity, 
        "energy_penalty": energy_penalty,
        "alive_bonus": alive_bonus
    }
    run_episode(env_name, reward_weights, episodes)



