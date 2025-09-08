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
        status.setdefault("fps", None)
    
    def _on_step(self) -> bool:
        steps_done = int(self.model.num_timesteps) # type: ignore
        status = self.status_dict[self.run_id] # this should be some sort of dictionary

        # Complete the crude FPS (over all the timesteps)
        dt = max(1e-6, time.time() - self._t0)
        fps = steps_done / dt

        # Pull Reward stats if available
        reward_last = None
        if "infos" in self.locals and self.locals["infos"]:
            ep_info = self.locals["infos"][-1].get("episode")
            if ep_info and "r" in ep_info:
                reward_last = ep_info["r"]

        reward_mean = None
        if getattr(self.model, "ep_info_buffer", None):
            try:
                reward_mean = float(np.mean([e["r"] for e in self.model.ep_info_buffer]))
            except Exception:
                pass
        
        # Update shared status (frontend can poll this)
        status.update({
            "steps_done": steps_done,
            "reward_last": reward_last,
            "reward_mean": reward_mean,
            "fps": fps,
        })

        if (steps_done - self._last_emit) >= self.every_n_steps:
            self._last_emit = steps_done
            if self.send_tick:
                try:
                    self.send_tick({
                        "type": "tick",
                        "run_id": self.run_id,
                        "step": steps_done,
                        "reward_last": reward_last,
                        "reward_mean": reward_mean,
                        "fps": fps,
                        "ts": time.time(),
                    })
                except Exception:
                    pass
            
            # Send a frame if it is requested
            if self.frame_fn and self.send_frame:
                try:
                    frame = self.frame_fn()
                    if frame is not None:
                        self.send_frame(frame) # Can encode this inside of the sender
                except Exception:
                    pass
        
        if status.get("stop", False):
            return False # allow stop flag that is triggered from other parts of the class

        return True



