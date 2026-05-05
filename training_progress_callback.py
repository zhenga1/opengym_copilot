# training_progress_callback.py

from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import time
import numpy as np

class TrainingProgressCallback(BaseCallback):
    """
    - Updates RUNS_TRAINING_STATUS[run_id] with steps_done, reward stats, etc.
    - Optionally renders/sends frames every N steps via a provided sender.
    - Can be combined with any existing user callback via CallbackList.
    """

    def __init__(
            self, 
            run_id:str,
            status_dict:dict,
            total_steps:int,
            every_n_steps: int = 20,
            frame_fn = None, # callable () -> np.uint8 [H, W, 3] (or None)
            send_frame = None, # callable (bytes) -> None / awaitable
            send_tick = None, # callable: (dict) -> None / awaitable
            stop_flag_key:str = "stop",
            ):
        super().__init__()
        self.run_id = run_id
        self.status_dict = status_dict
        self.total_steps = total_steps
        self.every_n_steps = max(1, every_n_steps) # so its not 0
        self.frame_fn = frame_fn
        self.send_frame = send_frame
        self.send_tick = send_tick
        self.stop_flag_key = stop_flag_key
        self._last_emit = 0
        self._t0 = time.time()
        ## Initialize the entry field for the status
        status_dict.setdefault(run_id, {})
        status = status_dict[run_id] # should be empty dic
        status.setdefault("status", "running")
        status.setdefault("model_path", None)
        status.setdefault("error", None)
        status.setdefault("total_steps", self.total_steps)
        status.setdefault("steps_done", 0)
        status.setdefault("reward_last", None)
        status.setdefault("reward_mean", None)
        status.setdefault("eval_reward", None)
        status.setdefault("reward_breakdown_last", {})
        status.setdefault("reward_breakdown_mean", {})
        status.setdefault("fps", None)

    @staticmethod
    def _extract_reward_breakdown(ep_info: dict | None) -> dict[str, float]:
        if not ep_info:
            return {}

        breakdown = {}
        for key, value in ep_info.items():
            if not key.startswith("reward_"):
                continue
            if key.startswith("reward_raw_"):
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                breakdown[key.removeprefix("reward_")] = float(value)

        if "total" not in breakdown and "r" in ep_info:
            breakdown["total"] = float(ep_info["r"])
        return breakdown
    
    def _on_step(self) -> bool:
        print("Executing On STEP from training_progress_callback")
        steps_done = int(self.model.num_timesteps) # type: ignore
        status = self.status_dict[self.run_id] # this should be some sort of dictionary

        # Complete the crude FPS (over all the timesteps)
        dt = max(1e-6, time.time() - self._t0)
        fps = steps_done / dt

        # Pull Reward stats if available
        reward_last = None
        reward_mean = None
        reward_breakdown_last = {}
        reward_breakdown_mean = {}
        if getattr(self.model, "ep_info_buffer", None):
            buffer_entries = list(self.model.ep_info_buffer)
            ep_rewards = [ep_info["r"] for ep_info in buffer_entries]
            reward_last = ep_rewards[-1] if ep_rewards else None
            reward_mean = np.mean(ep_rewards) if ep_rewards else None
            breakdown_buffer = [self._extract_reward_breakdown(ep_info) for ep_info in buffer_entries]
            reward_breakdown_last = breakdown_buffer[-1] if breakdown_buffer else {}
            breakdown_keys = {key for breakdown in breakdown_buffer for key in breakdown}
            reward_breakdown_mean = {
                key: float(np.mean([breakdown[key] for breakdown in breakdown_buffer if key in breakdown]))
                for key in breakdown_keys
            }
        
        # Update shared status (frontend can poll this)
        status.update({
            "steps_done": steps_done,
            "reward_last": reward_last,
            "reward_mean": reward_mean,
            "reward_breakdown_last": reward_breakdown_last,
            "reward_breakdown_mean": reward_breakdown_mean,
            "fps": fps,
        })
        
        # print(f"Preparing to send training progress callback with reward {reward_last} and mean reward {reward_mean}")
        # print(f"Current steps done {steps_done}, steps last emit {self._last_emit}, every_n_steps {self.every_n_steps}")
        if (steps_done - self._last_emit) >= self.every_n_steps:
            self._last_emit = steps_done
            if self.send_tick:
                try:
                    print(f"Sending tick from training_progress_callback with reward of {reward_last} and reward_mean of {reward_mean}")
                    self.send_tick({
                        "type": "tick",
                        "run_id": self.run_id,
                        "step": steps_done,
                        "reward": reward_last,
                        "reward_mean": reward_mean,
                        "eval_reward": status.get("eval_reward"),
                        "reward_breakdown": reward_breakdown_last,
                        "reward_breakdown_mean": reward_breakdown_mean,
                        "fps": fps,
                        "ts": time.time(),
                    }, self.run_id)

                except Exception as e:
                    print(f"Error sending tick from training_progress_callback: {e}")
                    pass
            
            # Send a frame if it is requested
            if self.frame_fn and self.send_frame:
                try:
                    frame = self.frame_fn()
                    if frame is not None:
                        print("Sending frame from training_progress_callback")
                        self.send_frame(frame, self.run_id) # Can encode this inside of the sender
                except Exception as e:
                    print(f"Error sending frame from training_progress_callback {e}")
                    pass
        
        if status.get("stop", False):
            return False # allow stop flag that is triggered from other parts of the class

        return True



