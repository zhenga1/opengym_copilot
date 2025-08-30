import math
import numpy as np

try:
    import gymnasium as gym   # new API
    GYMNASIUM = True
except ImportError:
    import gym                # classic API
    GYMNASIUM = False


class CartPoleDanceWrapper(gym.Wrapper):
    """
    Make CartPole move along a sinusoidal x_ref(t) while staying upright.
    Adds reward terms for position/velocity tracking + upright + action reg.

    Works with Gym or Gymnasium. By default it ADDS to native reward.
    Set w_native=0 to fully replace the native reward.
    """

    def __init__(
        self,
        env,
        A=1.0,            # amplitude in meters (CartPole threshold is ~2.4)
        f=0.25,           # frequency in Hz
        w_native=1.0,     # weight for native env reward (0 = replace)
        w_pos=-1.0,       # penalty scale for (x - x_ref)^2
        w_vel=-0.1,       # penalty scale for (xdot - v_ref)^2
        w_upright=-1.0,   # penalty scale for theta^2
        w_action=-0.001,  # small action^2 penalty
        augment_obs=True  # append x_ref & v_ref to observation
    ):
        super().__init__(env)
        self.A = float(A)
        self.f = float(f)
        self.w_native  = float(w_native)
        self.w_pos     = float(w_pos)
        self.w_vel     = float(w_vel)
        self.w_upright = float(w_upright)
        self.w_action  = float(w_action)
        self.augment_obs = augment_obs

        # CartPole uses tau=0.02s; try to read it, else default
        self.tau = getattr(env.unwrapped, "tau", 0.02)
        self.t = 0.0
        self.steps = 0

        # If augmenting obs, extend observation space by 2 for [x_ref, v_ref]
        if self.augment_obs:
            from gym.spaces import Box
            low  = np.concatenate([self.observation_space.low,  [-np.inf, -np.inf]])
            high = np.concatenate([self.observation_space.high, [ np.inf,  np.inf]])
            self.observation_space = Box(low=low, high=high, dtype=np.float32)

    def _ref(self, t):
        x_ref = self.A * math.sin(2*math.pi*self.f*t)
        v_ref = 2*math.pi*self.f*self.A * math.cos(2*math.pi*self.f*t)
        return x_ref, v_ref

    def reset(self, **kwargs):
        if GYMNASIUM:
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)
            info = {}

        self.t = 0.0
        self.steps = 0

        if self.augment_obs:
            x_ref, v_ref = self._ref(self.t)
            obs = np.concatenate([np.array(obs, dtype=np.float32),
                                  np.array([x_ref, v_ref], dtype=np.float32)])
        return (obs, info) if GYMNASIUM else obs

    def step(self, action):
        if GYMNASIUM:
            obs, native_r, terminated, truncated, info = self.env.step(action)
            done_flag = (terminated or truncated)
        else:
            obs, native_r, done_flag, info = self.env.step(action)

        # State = [x, x_dot, theta, theta_dot]
        x, xdot, theta, thetadot = obs

        # Time for current step (before increment)
        t = self.steps * self.tau
        x_ref, v_ref = self._ref(t)

        # Reward terms (quadratic tracking + upright + action reg)
        r_pos     = self.w_pos     * (x - x_ref)**2
        r_vel     = self.w_vel     * (xdot - v_ref)**2
        r_upright = self.w_upright * (theta**2)
        # CartPole has discrete actions {0,1}; map to {-1,+1} for a smoother penalty
        a = float(action if np.isscalar(action) else action[0])
        a_cont = -1.0 if int(a) == 0 else 1.0
        r_action = self.w_action * (a_cont**2)

        shaped = r_pos + r_vel + r_upright + r_action
        total_reward = self.w_native * native_r + shaped

        # Attach a breakdown for logging
        info = dict(info)
        info["dance_reward"] = {
            "native": float(native_r),
            "pos": float(r_pos),
            "vel": float(r_vel),
            "upright": float(r_upright),
            "action": float(r_action),
            "total": float(total_reward),
            "x_ref": float(x_ref),
            "v_ref": float(v_ref),
            "t": float(t),
            "step": int(self.steps),
        }

        self.steps += 1
        self.t += self.tau

        # Optionally append refs to observation so the policy knows the target
        if self.augment_obs:
            obs = np.concatenate([np.array(obs, dtype=np.float32),
                                  np.array([x_ref, v_ref], dtype=np.float32)])

        if GYMNASIUM:
            return obs, total_reward, terminated, truncated, info
        else:
            return obs, total_reward, done_flag, info
