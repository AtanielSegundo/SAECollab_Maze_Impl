"""
GPUMazeWrapper — drop-in replacement for MazeGymWrapper.

Key design:
  - Base state encodings stored as GPU lookup tables (rows*cols, enc_size)
  - Possible-actions mask stored as GPU lookup table (rows*cols, 4)
  - History buffers (last_states, last_actions) maintained as GPU tensors
  - CPU mirrors of history buffers kept in sync for zero-cost last_state_np
  - reset() / step() return GPU tensors with shape (1, state_size) — ready for act()
  - last_state_np property returns CPU numpy with NO GPU sync for COORDS/COORDS_NORM
    (ONE_HOT requires one GPU→CPU copy per step, unavoidable for large vectors)

Zero CPU→GPU transfers in the hot training loop.
"""

import torch
import numpy as np
from typing import Optional, Tuple


class GPUMazeWrapper:

    def __init__(
        self,
        maze,                                   # MazeEnv instance
        device,                                 # torch.device or str
        state_encoder=None,                     # StateEncoder enum value
        num_last_states: int = None,
        num_last_actions: int = None,
        possible_actions_feature: bool = False,
        visited_count: bool = False,
        **kwargs
    ):
        # Import here to avoid circular deps when used standalone
        from env.MazeEnv import Action, GridCell
        from env.MazeWrapper import StateEncoder

        self.maze      = maze
        self.device    = torch.device(device) if isinstance(device, str) else device
        self.rows      = maze.rows
        self.cols      = maze.cols
        self.file_path = maze.file_path
        self._StateEncoder = StateEncoder

        # Default encoder
        if state_encoder is None:
            state_encoder = StateEncoder.COORDS
        self._enc = state_encoder

        self._n_ls   = num_last_states  or 0
        self._n_la   = num_last_actions or 0
        self._use_pa = possible_actions_feature
        self._use_vc = visited_count

        self.action_size = len(list(Action))
        self.start_r     = int(maze.agent_start[0])
        self.start_c     = int(maze.agent_start[1])
        self.goal_r      = int(maze.agent_goal[0])
        self.goal_c      = int(maze.agent_goal[1])

        # ── Action metadata ───────────────────────────────────────────────
        # Action.delta = (dc, dr)  →  step: nr = r + delta[1], nc = c + delta[0]
        actions      = list(Action)
        self._adr    = [int(a.delta[1]) for a in actions]   # row deltas (CPU list)
        self._adc    = [int(a.delta[0]) for a in actions]   # col deltas (CPU list)
        # delta vectors for last_actions feature — stored as (dc, dr) tuple matching orig
        adv_np       = np.array([list(a.delta) for a in actions], dtype=np.float32)
        self._adv_gpu = torch.from_numpy(adv_np).to(self.device)   # (4, 2)
        self._adv_cpu = adv_np                                       # (4, 2) numpy

        # ── Grid (CPU only — single int lookup per step, trivial) ─────────
        self._grid = maze.grid          # numpy uint8
        self._rew  = {v.value:v.reward for v in list(GridCell)}

        # ── Base encoding lookup tables ───────────────────────────────────
        n = self.rows * self.cols
        if state_encoder == StateEncoder.COORDS:
            self.base_enc_size = 2
            enc_np = np.array(
                [[r, c] for r in range(self.rows) for c in range(self.cols)],
                dtype=np.float32
            )
        elif state_encoder == StateEncoder.COORDS_NORM:
            self.base_enc_size = 2
            enc_np = np.array(
                [[r / (self.rows - 1), c / (self.cols - 1)]
                 for r in range(self.rows) for c in range(self.cols)],
                dtype=np.float32
            )
        elif state_encoder == StateEncoder.ONE_HOT:
            self.base_enc_size = n
            enc_np = np.eye(n, dtype=np.float32)
        else:
            raise ValueError(f"Unknown StateEncoder: {state_encoder}")

        self._benc_gpu = torch.from_numpy(enc_np).to(self.device)   # (n, enc_size)
        self._benc_cpu = enc_np                                       # CPU mirror

        # ── Possible-actions lookup table ─────────────────────────────────
        if self._use_pa:
            masks = np.zeros((n, self.action_size), dtype=np.float32)
            for r in range(self.rows):
                for c in range(self.cols):
                    for ai, a in enumerate(actions):
                        nr = r + int(a.delta[1])
                        nc = c + int(a.delta[0])
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            if maze.grid[nr, nc] != 1:   # not WALL
                                masks[r * self.cols + c, ai] = 1.0
            self._pa_gpu = torch.from_numpy(masks).to(self.device)
            self._pa_cpu = masks

        # ── State size ────────────────────────────────────────────────────
        if state_encoder == StateEncoder.ONE_HOT:
            # ONE_HOT: history is blended into base encoding, not concatenated
            self.state_size = self.base_enc_size
        else:
            self.state_size = self.base_enc_size + 2 * self._n_ls

        self.state_size += 2 * self._n_la
        if self._use_pa:
            self.state_size += self.action_size

        # ── History buffers (GPU + CPU mirrors) ───────────────────────────
        if self._n_ls > 0:
            if state_encoder == StateEncoder.ONE_HOT:
                # Store indices; CPU mirror for numpy assembly
                self._ls_idx_gpu = torch.zeros(self._n_ls, dtype=torch.long,  device=self.device)
                self._ls_idx_cpu = np.zeros(self._n_ls, dtype=np.int64)
                # Weights for blending: 0.5^i where i=0 is oldest
                self._oh_w_gpu = torch.tensor(
                    [0.5**i for i in range(self._n_ls)],
                    dtype=torch.float32, device=self.device
                )
            else:
                self._ls_gpu = torch.zeros(self._n_ls, 2, dtype=torch.float32, device=self.device)
                self._ls_cpu = np.zeros((self._n_ls, 2), dtype=np.float32)
            self._ls_len = 0

        if self._n_la > 0:
            self._la_gpu = torch.zeros(self._n_la, 2, dtype=torch.float32, device=self.device)
            self._la_cpu = np.zeros((self._n_la, 2), dtype=np.float32)
            self._la_len = 0

        if self._use_vc:
            # Visited count lives on CPU (the `and` bug in original means it's always 0
            # anyway — matched here for correctness)
            self._vc = np.zeros(n, dtype=np.float32)

        # ── Pre-allocated output tensor (returned by reset/step) ──────────
        # Ping-pong buffers so caller can safely hold the previous tensor
        # while we write the next one.
        self._ping = torch.zeros(1, self.state_size, dtype=torch.float32, device=self.device)
        self._pong = torch.zeros(1, self.state_size, dtype=torch.float32, device=self.device)
        self._use_ping = True          # which buffer is "current"

        # ── Current position ──────────────────────────────────────────────
        self._r = self.start_r
        self._c = self.start_c

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _idx(self) -> int:
        return self._r * self.cols + self._c

    def _write_gpu(self, idx: int, buf: torch.Tensor) -> None:
        """Fill buf[0] with the full state vector using GPU lookups only."""
        SE   = self._StateEncoder
        base = self._benc_gpu[idx]           # (enc_size,)  — GPU

        if self._enc == SE.ONE_HOT:
            s = base.clone()
            if self._n_ls > 0 and self._ls_len > 0:
                n    = min(self._ls_len, self._n_ls)
                past = self._benc_gpu[self._ls_idx_gpu[:n]]   # (n, enc_size)
                w    = self._oh_w_gpu[:n].unsqueeze(1)         # (n, 1)
                s    = s + (past * w).sum(0)
            if self._use_pa:
                buf[0, :self.base_enc_size] = s
                buf[0, self.base_enc_size:] = self._pa_gpu[idx]
            else:
                buf[0] = s
            return

        # COORDS / COORDS_NORM — slice-write into pre-allocated buf[0]
        offset = self.base_enc_size
        buf[0, :offset] = base

        if self._n_ls > 0:
            end = offset + self._n_ls * 2
            buf[0, offset:end] = self._ls_gpu.reshape(-1)
            offset = end

        if self._n_la > 0:
            end = offset + self._n_la * 2
            buf[0, offset:end] = self._la_gpu.reshape(-1)
            offset = end

        if self._use_pa:
            buf[0, offset:] = self._pa_gpu[idx]

    def _build_cpu(self, idx: int) -> np.ndarray:
        """Build state as numpy from CPU-resident data — no GPU sync for COORDS."""
        SE = self._StateEncoder

        if self._enc == SE.ONE_HOT:
            # ONE_HOT: need GPU indices, one small sync
            s = self._benc_cpu[idx].copy()
            if self._n_ls > 0 and self._ls_len > 0:
                n = min(self._ls_len, self._n_ls)
                for i in range(n):
                    s += self._benc_cpu[self._ls_idx_cpu[i]] * (0.5 ** i)
            if self._use_pa:
                return np.concatenate([s, self._pa_cpu[idx]])
            return s

        # COORDS / COORDS_NORM — pure CPU, zero GPU sync
        parts = [self._benc_cpu[idx]]
        if self._n_ls > 0:
            parts.append(self._ls_cpu.reshape(-1))
        if self._n_la > 0:
            parts.append(self._la_cpu.reshape(-1))
        if self._use_pa:
            parts.append(self._pa_cpu[idx])
        return np.concatenate(parts) if len(parts) > 1 else parts[0].copy()

    def _next_buf(self) -> torch.Tensor:
        """Return the *other* ping-pong buffer (to write into next)."""
        if self._use_ping:
            return self._pong
        return self._ping

    def _swap(self) -> torch.Tensor:
        """Swap active buffer and return the newly active one."""
        self._use_ping = not self._use_ping
        return self._ping if self._use_ping else self._pong

    # ──────────────────────────────────────────────────────────────────────
    # Public API  (matches MazeGymWrapper interface)
    # ──────────────────────────────────────────────────────────────────────

    @property
    def last_state_np(self) -> np.ndarray:
        """
        Numpy (state_size,) of the current state.
        For COORDS/COORDS_NORM: pure CPU, zero GPU sync.
        For ONE_HOT: reads CPU-mirrored indices, still no CUDA sync.
        """
        return self._build_cpu(self._idx())

    def reset(self) -> torch.Tensor:
        """Returns GPU tensor shape (1, state_size). Resets all history."""
        self._r = self.start_r
        self._c = self.start_c

        if self._n_ls > 0:
            if self._enc == self._StateEncoder.ONE_HOT:
                self._ls_idx_gpu.zero_()
                self._ls_idx_cpu[:] = 0
            else:
                self._ls_gpu.zero_()
                self._ls_cpu[:] = 0.0
            self._ls_len = 0

        if self._n_la > 0:
            self._la_gpu.zero_()
            self._la_cpu[:] = 0.0
            self._la_len = 0

        if self._use_vc:
            self._vc[:] = 0.0

        buf = self._next_buf()
        self._write_gpu(self._idx(), buf)
        return self._swap()

    def step(self, action_idx: int) -> Tuple[torch.Tensor, float, bool, dict]:
        """
        Returns (state_gpu, reward, done, info).
        state_gpu: shape (1, state_size) on self.device — no CPU→GPU transfer.
        """
        r, c = self._r, self._c
        nr   = r + self._adr[action_idx]
        nc   = c + self._adc[action_idx]

        # ── Transition logic (pure CPU integer arithmetic) ────────────────
        oob = (nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols)

        if oob:
            reward  = -0.5
            nr, nc  = r, c
            is_goal = False
        else:
            cell    = int(self._grid[nr, nc])
            is_goal = (cell == 2)
            if cell == 1:                          # WALL
                reward = -0.5
                if not self.maze.pass_through_walls:
                    nr, nc = r, c
            else:
                reward = self._rew.get(cell, -0.01)

        next_idx = nr * self.cols + nc

        # Visited count — matches original `and` (not `or`) semantics
        if self._use_vc:
            if nr != r and nc != c:
                self._vc[next_idx] += 1
            reward *= float(1.0 + self._vc[next_idx])

        # ── Update history buffers (GPU + CPU mirrors) ────────────────────
        # Deque semantics: index 0 = oldest, -1 = newest. Shift left, append at end.
        if self._n_ls > 0:
            if self._ls_len < self._n_ls:
                self._ls_len += 1
            if self._enc == self._StateEncoder.ONE_HOT:
                # GPU shift
                if self._n_ls > 1:
                    self._ls_idx_gpu[:-1].copy_(self._ls_idx_gpu[1:].clone())
                self._ls_idx_gpu[-1] = next_idx
                # CPU mirror shift
                if self._n_ls > 1:
                    self._ls_idx_cpu[:-1] = self._ls_idx_cpu[1:]
                self._ls_idx_cpu[-1] = next_idx
            else:
                if self._n_ls > 1:
                    self._ls_gpu[:-1].copy_(self._ls_gpu[1:].clone())
                    self._ls_cpu[:-1] = self._ls_cpu[1:]
                self._ls_gpu[-1]  = self._benc_gpu[next_idx]
                self._ls_cpu[-1]  = self._benc_cpu[next_idx]

        if self._n_la > 0:
            if self._la_len < self._n_la:
                self._la_len += 1
            if self._n_la > 1:
                self._la_gpu[:-1].copy_(self._la_gpu[1:].clone())
                self._la_cpu[:-1] = self._la_cpu[1:]
            self._la_gpu[-1] = self._adv_gpu[action_idx]
            self._la_cpu[-1] = self._adv_cpu[action_idx]

        self._r = nr
        self._c = nc

        # ── Assemble output tensor ────────────────────────────────────────
        buf = self._next_buf()
        self._write_gpu(next_idx, buf)
        return self._swap(), float(reward), bool(is_goal), {"raw_ns": (nr, nc)}

    def isGoal(self, rc: tuple) -> bool:
        return rc[0] == self.goal_r and rc[1] == self.goal_c

    # ── Compatibility properties ──────────────────────────────────────────
    @property
    def start(self): return (self.start_r, self.start_c)

    @property
    def goal(self):  return (self.goal_r, self.goal_c)