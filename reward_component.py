def compute_reward_components(obs, action, next_obs):
    return {
        "forward_velocity": next_obs[0],
        "energy_penalty": -abs(action).sum()
    }