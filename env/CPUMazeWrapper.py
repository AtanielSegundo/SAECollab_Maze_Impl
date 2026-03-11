from env.MazeWrapper import MazeGymWrapper
import numpy as np
import torch

class CPUMazeWrapperAdapter:
    """
    Wraps MazeGymWrapper to match GPUMazeWrapper's training-loop interface:
      - reset() / step() return GPU tensors of shape (1, state_size)
      - last_state_np property returns the current state as a CPU numpy array (state_size,)
        with zero GPU sync (tensor was built from numpy in the first place)

    This lets all training functions written against GPUMazeWrapper work with
    MazeGymWrapper without any changes to those functions.
    """

    def __init__(self, gym_env: MazeGymWrapper, device: torch.device):
        self._env        = gym_env
        self.device      = device
        self._last_np: np.ndarray = None

        # Proxy attributes expected by training code
        self.state_size  = gym_env.state_size
        self.action_size = gym_env.action_size
        self.rows        = gym_env.rows
        self.cols        = gym_env.cols
        self.start       = gym_env.start
        self.goal        = gym_env.goal
        self.maze        = gym_env.maze

    # ── last_state_np ─────────────────────────────────────────────────────
    @property
    def last_state_np(self) -> np.ndarray:
        """CPU numpy (state_size,) — cached from the most recent reset/step."""
        return self._last_np

    # ── helpers ───────────────────────────────────────────────────────────
    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(
            np.asarray(arr, dtype=np.float32)
        ).unsqueeze(0).to(self.device)

    # ── public API ────────────────────────────────────────────────────────
    def reset(self) -> torch.Tensor:
        """Returns GPU tensor (1, state_size). Resets all history."""
        np_state       = self._env.reset()          # (state_size,) numpy
        self._last_np  = np_state
        return self._to_tensor(np_state)

    def step(self, action_idx: int):
        """Returns (gpu_tensor (1,state_size), reward, done, info)."""
        np_state, reward, done, info = self._env.step(action_idx)
        self._last_np = np_state
        return self._to_tensor(np_state), float(reward), bool(done), info

    def isGoal(self, rc: tuple) -> bool:
        return self._env.isGoal(rc)
