from typing import *

import os
import sys
import csv
import json
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn

from env.MazeEnv     import MazeEnv, Action, GridCell
from env.MazeWrapper import StateEncoder
from models.QModels  import (
    TorchDDQN, ReservedSAECollabDDQN, NewLayerCfg, exp_decay_factor_to,
)

# ablation.config / ablation.training sao importaveis sem efeito colateral.
# (ablation.experiments NAO e: ele chama multiprocessing.set_start_method no
#  import. Nao importe nada de la aqui.)
from ablation.config             import ModelArch, LayerInsertionType, LayerModeType
from ablation.training           import gen_concrete_arch
from StackedCollab.collabNet     import LayersConfig, MutationMode

"""
O objetivo desse projeto e desenvolver um ambiente onde
o agente de RL aprende a solucionar nao apenas um labirinto
mas chegar a qualquer Goal (x_final,y_final) a partir de
qualquer (x_inicial,y_inicial)

-----------------------------------------------------------------------------
PROTOTIPO STANDALONE — nao importa nada de ablation/ e nao modifica env/.
Reusa apenas MazeEnv (leitura do .maze) e TorchDDQN (agente).

Formulacao: UVFA (Schaul et al. 2015) — Q(s, g, a) em vez de Q(s, a).
            HER  (Andrychowicz et al. 2017), estrategia 'future'.

LAYOUT DA OBSERVACAO  (a fatia do goal e contigua e fica NO FIM):

    [ ---------------- prefixo ---------------- | ----- goal slice ----- ]
    [ enc(pos) | last_states | last_actions | pa | enc(goal) | dr | dc   ]
    0                                        goal_off              state_size

    O prefixo depende so da posicao/historico -> e INVARIANTE ao goal.
    Por isso o relabel do HER precisa reescrever apenas obs[:, goal_off:].

MODELO DE RECOMPENSA (auto-contido, reconstruivel a partir de indices):

    next == goal             -> +1.0, done=True
    next == cur (parede/oob) -> -0.5
    caso contrario           -> -0.01
    + shaping (opcional)     -> gamma * Phi_g(next) - Phi_g(cur)
      com Phi_g(s) = -manhattan(s, g) / max_dist   em [-1, 0]

    Note que (cur_idx, next_idx, goal_idx) determinam reward e done
    COMPLETAMENTE. Essa e a propriedade que torna o HER possivel: ao trocar o
    goal, a recompensa e recalculada exatamente, sem re-simular o episodio.

    Consequencia: `visited_count` (que escala o reward pelo numero de visitas)
    NAO e suportado aqui — quebraria essa reconstrutibilidade.

USO:
    python dynamic_maze_start_goal.py --selftest
    python dynamic_maze_start_goal.py -m mazes/small_eg.maze -s 333
    python dynamic_maze_start_goal.py -m mazes/small_eg.maze --no_her
    python dynamic_maze_start_goal.py -h
"""


WALL = GridCell.WALL.value          # 1

# Deltas de linha/coluna por indice de acao, na mesma convencao do
# GPUMazeWrapper: Action.delta = (dc, dr) -> nr = r + delta[1], nc = c + delta[0]
_ACTIONS = list(Action)
_ADR     = np.array([int(a.delta[1]) for a in _ACTIONS], dtype=np.int32)
_ADC     = np.array([int(a.delta[0]) for a in _ACTIONS], dtype=np.int32)
_ADV     = np.array([list(a.delta)   for a in _ACTIONS], dtype=np.float32)  # (4, 2)

TOPOLOGY_CACHE_DIR = "./.dyn_topology_cache"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Topologia — vizinhanca, alcancabilidade e distancias otimas (BFS all-pairs)
# ─────────────────────────────────────────────────────────────────────────────

class MazeTopology:
    """Pre-computa tudo que e funcao apenas das PAREDES (nao do start/goal).

    - `nbr[i, a]`  -> indice flat do vizinho de i pela acao a, ou -1 se for
                      parede / fora do grid.
    - `nav`        -> indices flat das celulas navegaveis (tudo != WALL,
                      incluindo as celulas marcadas START(9) e GOAL(2), que
                      aqui perdem seu significado especial).
    - `dist[i, j]` -> menor numero de passos de i ate j, ou -1 se inalcancavel.
                      E o comprimento OTIMO do caminho: serve de ground truth
                      para a razao de otimalidade, para o curriculo e para o
                      teste de alcancabilidade.

    A matriz de distancias depende so do arquivo .maze, entao e cacheada em
    disco. O BFS por fonte e vetorizado por fronteira (numpy), nao no-a-no.

    Nota: nao usamos models/AStar.py como ground truth porque ele compara
    `grid[...] == GridCell.WALL` (uint8 vs Enum, sempre False) e usa
    `maze.agent_goal` na heuristica mesmo quando recebe `alt_end`.
    """

    def __init__(self, maze: MazeEnv, cache: bool = True, verbose: bool = True):
        self.rows = int(maze.rows)
        self.cols = int(maze.cols)
        self.n    = self.rows * self.cols
        self.grid = np.asarray(maze.grid)
        self.tag  = maze.tag

        flat_grid     = self.grid.reshape(-1)
        self.walkable = (flat_grid != WALL)                  # bool (n,)
        self.nav      = np.flatnonzero(self.walkable).astype(np.int32)

        self.nbr  = self._build_neighbours()
        self.dist = self._load_or_build_distances(cache, verbose)

    # ── construcao ───────────────────────────────────────────────────────────
    def _build_neighbours(self) -> np.ndarray:
        nbr = np.full((self.n, len(_ACTIONS)), -1, dtype=np.int32)
        for r in range(self.rows):
            for c in range(self.cols):
                i = r * self.cols + c
                if self.grid[r, c] == WALL:
                    continue
                for a in range(len(_ACTIONS)):
                    nr = r + int(_ADR[a])
                    nc = c + int(_ADC[a])
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        if self.grid[nr, nc] != WALL:
                            nbr[i, a] = nr * self.cols + nc
        return nbr

    def _bfs_from(self, src: int) -> np.ndarray:
        """BFS por fronteira. Retorna (n,) int32 com -1 para inalcancavel."""
        dist      = np.full(self.n, -1, dtype=np.int32)
        dist[src] = 0
        frontier  = np.array([src], dtype=np.int32)
        d = 0
        while frontier.size:
            d += 1
            cand = self.nbr[frontier].reshape(-1)      # (k*4,)
            cand = cand[cand >= 0]
            if cand.size == 0:
                break
            cand = np.unique(cand)
            cand = cand[dist[cand] < 0]
            if cand.size == 0:
                break
            dist[cand] = d
            frontier   = cand
        return dist

    def _cache_path(self) -> str:
        return os.path.join(
            TOPOLOGY_CACHE_DIR, f"{self.tag}_{self.rows}x{self.cols}_dist.npy"
        )

    def _load_or_build_distances(self, cache: bool, verbose: bool) -> np.ndarray:
        path = self._cache_path()
        if cache and os.path.exists(path):
            try:
                d = np.load(path)
                if d.shape == (self.n, self.n):
                    if verbose:
                        print(f"[TOPO] distancias carregadas do cache: {path}")
                    return d
                print(f"[TOPO][WARN] cache com shape errado, recomputando: {path}")
            except Exception as e:
                print(f"[TOPO][WARN] falha lendo cache ({e}), recomputando")

        if verbose:
            print(f"[TOPO] BFS all-pairs sobre {self.nav.size} celulas navegaveis...")
        t0 = time.perf_counter()
        # int16 basta: a maior distancia possivel e << 32767 para estes mazes.
        dist = np.full((self.n, self.n), -1, dtype=np.int16)
        for k, src in enumerate(self.nav):
            dist[src] = self._bfs_from(int(src)).astype(np.int16)
            if verbose and self.nav.size > 200 and k and k % 200 == 0:
                print(f"[TOPO]   {k}/{self.nav.size}")
        if verbose:
            print(f"[TOPO] concluido em {time.perf_counter() - t0:.2f}s")

        if cache:
            os.makedirs(TOPOLOGY_CACHE_DIR, exist_ok=True)
            np.save(path, dist)
        return dist

    # ── consultas ────────────────────────────────────────────────────────────
    def rc(self, idx: int) -> Tuple[int, int]:
        return int(idx) // self.cols, int(idx) % self.cols

    def idx(self, r: int, c: int) -> int:
        return int(r) * self.cols + int(c)

    def optimal(self, start_idx: int, goal_idx: int) -> int:
        return int(self.dist[start_idx, goal_idx])

    def worst_optimal(self) -> int:
        return int(self.dist[np.ix_(self.nav, self.nav)].max())

    def n_valid_pairs(self) -> int:
        sub = self.dist[np.ix_(self.nav, self.nav)]
        return int((sub > 0).sum())


# ─────────────────────────────────────────────────────────────────────────────
# 2. Ambiente goal-conditioned
# ─────────────────────────────────────────────────────────────────────────────

class DynamicGoalMazeEnv:
    """MazeEnv com start e goal definidos POR EPISODIO, e o goal DENTRO da
    observacao.

    Diferencas centrais em relacao a MazeWrapper / GPUMazeWrapper:
      - `reset(start_idx, goal_idx)` recebe o par; nao le as celulas 9 / 2.
      - o objetivo e comparado por COORDENADA, nao por tipo de celula.
      - o potencial Phi do reward shaping segue o goal do episodio.
      - a observacao ganha uma fatia final [enc(goal) | dr | dc].
    """

    R_GOAL = 1.0
    R_WALL = -0.5
    R_STEP = -0.01

    def __init__(
        self,
        maze: MazeEnv,
        topo: MazeTopology,
        state_encoder: StateEncoder = StateEncoder.COORDS_NORM,
        num_last_states: int        = 0,
        num_last_actions: int       = 0,
        possible_actions_feature: bool = False,
        use_shaping: bool           = True,
        shaping_gamma: float        = 0.99,
        device: Union[str, torch.device] = "cpu",
    ):
        if getattr(maze, "pass_through_walls", False):
            raise ValueError(
                "pass_through_walls=True quebra a deteccao de colisao usada pelo "
                "HER (next_idx == cur_idx <=> bateu na parede)."
            )

        self.maze   = maze
        self.topo   = topo
        self.rows   = topo.rows
        self.cols   = topo.cols
        self.n      = topo.n
        self.device = torch.device(device)

        self._enc     = state_encoder
        self._blended = state_encoder in (StateEncoder.ONE_HOT, StateEncoder.MULTI_HOT)
        self._n_ls    = int(num_last_states  or 0)
        self._n_la    = int(num_last_actions or 0)
        self._use_pa  = bool(possible_actions_feature)

        self.action_size   = len(_ACTIONS)
        self.use_shaping   = bool(use_shaping)
        self.shaping_gamma = float(shaping_gamma)

        # ── tabelas de encoding (numpy, CPU) ────────────────────────────────
        self._benc         = self._build_base_encoding(state_encoder)
        self.base_enc_size = self._benc.shape[1]

        if self._use_pa:
            self._pa = (topo.nbr >= 0).astype(np.float32)      # (n, 4)

        # ── layout do vetor de observacao ───────────────────────────────────
        prefix = self.base_enc_size
        if not self._blended:
            # ONE_HOT / MULTI_HOT mesclam o historico na base (peso 0.5^i),
            # como faz o GPUMazeWrapper; os demais concatenam.
            prefix += 2 * self._n_ls
        prefix += 2 * self._n_la
        if self._use_pa:
            prefix += self.action_size

        self.goal_off     = prefix
        self.goal_enc_off = prefix
        self.delta_off    = prefix + self.base_enc_size
        self.state_size   = self.delta_off + 2
        self.goal_slice   = slice(self.goal_off, self.state_size)

        # ── normalizadores ──────────────────────────────────────────────────
        self._rden     = float(max(1, self.rows - 1))
        self._cden     = float(max(1, self.cols - 1))
        self._max_dist = float(max(1, (self.rows - 1) + (self.cols - 1)))

        # ── buffers de historico (semantica de deque: [0] = mais antigo) ────
        self._ls_idx = np.zeros(max(self._n_ls, 1), dtype=np.int64)     # blended
        self._ls_enc = np.zeros((max(self._n_ls, 1), 2), dtype=np.float32)
        self._ls_len = 0
        self._la     = np.zeros((max(self._n_la, 1), 2), dtype=np.float32)
        self._oh_w   = np.array([0.5 ** i for i in range(max(self._n_ls, 1))],
                                dtype=np.float32)

        # ── estado corrente ─────────────────────────────────────────────────
        self._obs      = np.zeros(self.state_size, dtype=np.float32)
        self.cur_idx   = 0
        self.goal_idx  = 0
        self.start_idx = 0

    # ── encodings base ───────────────────────────────────────────────────────
    def _build_base_encoding(self, enc: StateEncoder) -> np.ndarray:
        n, R, C = self.n, self.rows, self.cols
        if enc is StateEncoder.COORDS:
            return np.array([[r, c] for r in range(R) for c in range(C)],
                            dtype=np.float32)
        if enc is StateEncoder.COORDS_NORM:
            rd, cd = max(1, R - 1), max(1, C - 1)
            return np.array([[r / rd, c / cd] for r in range(R) for c in range(C)],
                            dtype=np.float32)
        if enc is StateEncoder.ONE_HOT:
            return np.eye(n, dtype=np.float32)
        if enc is StateEncoder.MULTI_HOT:
            e = np.zeros((n, R + C), dtype=np.float32)
            for r in range(R):
                for c in range(C):
                    e[r * C + c, r]     = 1.0
                    e[r * C + c, R + c] = 1.0
            return e
        raise ValueError(f"StateEncoder nao suportado: {enc}")

    # ── potencial / recompensa ───────────────────────────────────────────────
    def phi(self, idx, goal_idx):
        """Phi_g(s) = -manhattan(s, g) / max_dist. Vetorizado.

        Phi_g(g) == 0 por construcao, o que satisfaz a condicao de terminal
        exigida pelo potential-based shaping (Ng et al. 1999) e garante que a
        politica otima nao muda.
        """
        idx  = np.asarray(idx)
        goal = np.asarray(goal_idx)
        r,  c  = idx  // self.cols, idx  % self.cols
        gr, gc = goal // self.cols, goal % self.cols
        return -(np.abs(r - gr) + np.abs(c - gc)).astype(np.float32) / self._max_dist

    def transition_reward(self, cur_idx, next_idx, goal_idx):
        """Reward e done a partir de (cur, next, goal). Vetorizado.

        E o nucleo do HER: como nao depende de nada alem desses tres indices,
        trocar o goal permite recalcular a recompensa exatamente.
        """
        cur  = np.asarray(cur_idx)
        nxt  = np.asarray(next_idx)
        goal = np.asarray(goal_idx)

        reached = (nxt == goal)
        blocked = (nxt == cur)          # so ocorre em parede / fora do grid

        reward = np.where(
            reached, self.R_GOAL,
            np.where(blocked, self.R_WALL, self.R_STEP)
        ).astype(np.float32)

        if self.use_shaping:
            reward = reward + (
                self.shaping_gamma * self.phi(nxt, goal) - self.phi(cur, goal)
            )
        return reward.astype(np.float32), reached

    # ── montagem da observacao ───────────────────────────────────────────────
    def _write_prefix(self, out: np.ndarray, idx: int) -> None:
        """Escreve out[:goal_off]. Depende so de posicao/historico."""
        if self._blended:
            s = self._benc[idx].copy()
            if self._n_ls > 0 and self._ls_len > 0:
                k = min(self._ls_len, self._n_ls)
                s += (self._benc[self._ls_idx[:k]] * self._oh_w[:k, None]).sum(0)
            out[:self.base_enc_size] = s
            off = self.base_enc_size
        else:
            out[:self.base_enc_size] = self._benc[idx]
            off = self.base_enc_size
            if self._n_ls > 0:
                out[off:off + 2 * self._n_ls] = self._ls_enc.reshape(-1)
                off += 2 * self._n_ls

        if self._n_la > 0:
            out[off:off + 2 * self._n_la] = self._la.reshape(-1)
            off += 2 * self._n_la

        if self._use_pa:
            out[off:off + self.action_size] = self._pa[idx]

    def write_goal_slice(self, obs: np.ndarray, pos_idx, goal_idx) -> None:
        """Reescreve obs[..., goal_off:] in-place.

        Aceita obs 1-D (state_size,) com escalares, ou 2-D (N, state_size) com
        arrays. E a UNICA porta pela qual o goal entra na observacao — por isso
        o relabel do HER e exato por construcao.
        """
        ge, de = self.goal_enc_off, self.delta_off
        pos    = np.asarray(pos_idx)
        goal   = np.asarray(goal_idx)

        r,  c  = pos  // self.cols, pos  % self.cols
        gr, gc = goal // self.cols, goal % self.cols

        if obs.ndim == 1:
            obs[ge:de]  = self._benc[int(goal)]
            obs[de]     = (gr - r) / self._rden
            obs[de + 1] = (gc - c) / self._cden
        else:
            obs[:, ge:de]  = self._benc[goal]
            obs[:, de]     = (gr - r) / self._rden
            obs[:, de + 1] = (gc - c) / self._cden

    def build_obs(self, pos_idx: int, goal_idx: int) -> np.ndarray:
        """Observacao completa SEM historico (usada pelo selftest)."""
        out = np.zeros(self.state_size, dtype=np.float32)
        out[:self.base_enc_size] = self._benc[pos_idx]
        if self._use_pa:
            off = self.goal_off - self.action_size
            out[off:off + self.action_size] = self._pa[pos_idx]
        self.write_goal_slice(out, pos_idx, goal_idx)
        return out

    def _tensor(self) -> torch.Tensor:
        return torch.from_numpy(self._obs).unsqueeze(0).to(self.device)

    # ── API do ambiente ──────────────────────────────────────────────────────
    def reset(self, start_idx: int, goal_idx: int) -> torch.Tensor:
        if start_idx == goal_idx:
            raise ValueError("start_idx == goal_idx: episodio degenerado")
        self.start_idx = int(start_idx)
        self.cur_idx   = int(start_idx)
        self.goal_idx  = int(goal_idx)

        self._ls_len    = 0
        self._ls_idx[:] = 0
        self._ls_enc[:] = 0.0
        self._la[:]     = 0.0

        self._obs[:] = 0.0
        self._write_prefix(self._obs, self.cur_idx)
        self.write_goal_slice(self._obs, self.cur_idx, self.goal_idx)
        return self._tensor()

    def step(self, action_idx: int) -> Tuple[torch.Tensor, float, bool, dict]:
        cur = self.cur_idx
        nbr = int(self.topo.nbr[cur, action_idx])
        nxt = cur if nbr < 0 else nbr          # -1 => parede ou fora do grid

        reward, done = self.transition_reward(cur, nxt, self.goal_idx)
        reward, done = float(reward), bool(done)

        # historico (deque: shift para a esquerda, insere no fim)
        if self._n_ls > 0:
            self._ls_len = min(self._ls_len + 1, self._n_ls)
            if self._n_ls > 1:
                self._ls_idx[:-1] = self._ls_idx[1:]
                self._ls_enc[:-1] = self._ls_enc[1:]
            self._ls_idx[-1] = nxt
            if not self._blended:
                self._ls_enc[-1] = self._benc[nxt]
        if self._n_la > 0:
            if self._n_la > 1:
                self._la[:-1] = self._la[1:]
            self._la[-1] = _ADV[action_idx]

        self.cur_idx = nxt
        self._obs[:] = 0.0
        self._write_prefix(self._obs, nxt)
        self.write_goal_slice(self._obs, nxt, self.goal_idx)

        info = {"raw_ns": self.topo.rc(nxt), "cur_idx": cur, "next_idx": nxt}
        return self._tensor(), reward, done, info

    def isGoal(self, rc) -> bool:
        if isinstance(rc, (tuple, list)):
            return self.topo.idx(rc[0], rc[1]) == self.goal_idx
        return int(rc) == self.goal_idx


# ─────────────────────────────────────────────────────────────────────────────
# 3. Amostragem de pares (start, goal)
# ─────────────────────────────────────────────────────────────────────────────

class PairSampler:
    """Amostra pares (start, goal) navegaveis, alcancaveis e distintos.

    `exclude` recebe o conjunto de avaliacao para que ele seja um held-out de
    verdade: pares de teste nunca aparecem no treino. Sem isso, "taxa de
    sucesso" mede memorizacao, nao generalizacao.

    `max_dist` habilita curriculo por distancia BFS (opcional; com HER ligado
    normalmente nao e necessario, mas ajuda a diagnosticar).
    """

    def __init__(self, topo: MazeTopology, rng: np.random.Generator,
                 exclude: Optional[Set[int]] = None,
                 min_dist: int = 1, max_dist: Optional[int] = None):
        self.topo     = topo
        self.rng      = rng
        self.nav      = topo.nav
        self.exclude  = exclude or set()
        self.min_dist = int(min_dist)
        self.max_dist = max_dist

    def key(self, s: int, g: int) -> int:
        return int(s) * self.topo.n + int(g)

    def sample(self, max_tries: int = 10_000) -> Tuple[int, int]:
        for _ in range(max_tries):
            s = int(self.rng.choice(self.nav))
            g = int(self.rng.choice(self.nav))
            if s == g:
                continue
            d = int(self.topo.dist[s, g])
            if d < self.min_dist:
                continue
            if self.max_dist is not None and d > self.max_dist:
                continue
            if self.key(s, g) in self.exclude:
                continue
            return s, g
        raise RuntimeError(
            f"Nenhum par valido apos {max_tries} tentativas "
            f"(min_dist={self.min_dist}, max_dist={self.max_dist}). "
            f"Curriculo apertado demais ou maze desconexo?"
        )


def make_eval_set(topo: MazeTopology, rng: np.random.Generator,
                  n_pairs: int) -> np.ndarray:
    """Conjunto held-out fixo de pares (start, goal). Retorna (n, 2) int32."""
    sampler = PairSampler(topo, rng)
    seen: Set[int] = set()
    pairs: List[Tuple[int, int]] = []
    guard = 0
    while len(pairs) < n_pairs and guard < n_pairs * 200:
        guard += 1
        s, g = sampler.sample()
        k = sampler.key(s, g)
        if k in seen:
            continue
        seen.add(k)
        pairs.append((s, g))
    if len(pairs) < n_pairs:
        print(f"[WARN] eval set com {len(pairs)}/{n_pairs} pares (maze pequeno?)")
    return np.array(pairs, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# 4. HER — relabel 'future'
# ─────────────────────────────────────────────────────────────────────────────

class EpisodeTape:
    """Buffer de um episodio. Guarda o vetor completo de observacao (com o goal
    real) mais os indices de celula, que sao tudo que o relabel precisa."""

    def __init__(self, max_steps: int, state_size: int):
        self.obs      = np.zeros((max_steps, state_size), dtype=np.float32)
        self.next_obs = np.zeros((max_steps, state_size), dtype=np.float32)
        self.action   = np.zeros(max_steps, dtype=np.int64)
        self.reward   = np.zeros(max_steps, dtype=np.float32)
        self.done     = np.zeros(max_steps, dtype=bool)
        self.cur_idx  = np.zeros(max_steps, dtype=np.int64)
        self.next_idx = np.zeros(max_steps, dtype=np.int64)
        self.t = 0

    def reset(self) -> None:
        self.t = 0

    def add(self, obs, action, reward, next_obs, done, cur_idx, next_idx) -> None:
        t = self.t
        self.obs[t]      = obs
        self.next_obs[t] = next_obs
        self.action[t]   = action
        self.reward[t]   = reward
        self.done[t]     = done
        self.cur_idx[t]  = cur_idx
        self.next_idx[t] = next_idx
        self.t += 1

    def __len__(self) -> int:
        return self.t


def sample_future_goals(tape: EpisodeTape, k: int, rng: np.random.Generator):
    """Estrategia 'future' (Andrychowicz et al. 2017).

    Para cada transicao t, sorteia k indices j em [t, T-1] e adota
    g' = next_idx[j] como goal ficticio. Com j == t o proprio passo vira um
    sucesso — e essa injecao de recompensa positiva que destrava o aprendizado
    quando o goal real quase nunca e alcancado por exploracao.

    Retorna (t_idx, goals) ou None. A construcao das transicoes em si fica a
    cargo de build_nstep, que e a mesma para goal real e goal ficticio.
    """
    T = len(tape)
    if T == 0 or k <= 0:
        return None

    t_idx = np.repeat(np.arange(T, dtype=np.int64), k)              # (T*k,)
    span  = (T - t_idx).astype(np.float64)                          # >= 1
    j_idx = t_idx + np.floor(rng.random(t_idx.size) * span).astype(np.int64)
    j_idx = np.minimum(j_idx, T - 1)

    g_new = tape.next_idx[j_idx]

    # Descarta relabels degenerados: o agente ja estava no goal ficticio ANTES
    # de agir, o que produziria uma transicao que o episodio real nunca teria.
    keep = (tape.cur_idx[t_idx] != g_new)
    if not keep.any():
        return None
    return t_idx[keep], g_new[keep]


def build_nstep(env: DynamicGoalMazeEnv, tape: EpisodeTape,
                t_idx: np.ndarray, goals: np.ndarray,
                n: int, gamma: float):
    """Constroi transicoes n-step para as janelas que comecam em `t_idx`, cada
    uma sob o goal correspondente em `goals`. Vetorizado sobre as M janelas.

        G_t = sum_{i<L} gamma^i * r_{t+i}   +   [bootstrap de s_{t+L}]

    O bootstrap NAO entra aqui: quem o aplica e o learn() do agente, via
    `target = R + agent.gamma * Q(next) * (1 - done)`. Por isso o treino passa
    `gamma ** n` como gamma DO AGENTE — ver train_dynamic().

    Terminacao sob o goal da janela
    -------------------------------
    Sob um goal ficticio g', o episodio termina na PRIMEIRA vez que o agente
    pisa em g' — que pode ser antes do fim da janela. Somar recompensas depois
    disso contaria passos que, sob g', nunca teriam acontecido. Por isso L e o
    indice da primeira terminacao dentro da janela, nao n fixo.

    Janelas cortadas pelo cap de passos (sem terminal e com menos de n passos
    disponiveis) sao DESCARTADAS: o bootstrap delas exigiria gamma^L com L < n,
    e o agente so tem um gamma. Custa no maximo n-1 transicoes por episodio
    truncado — preferivel a introduzir vies no alvo.

    Com n == 1 esta funcao reproduz exatamente a transicao 1-step original
    (invariante T5/T9 do selftest).

    Retorna (obs, actions, returns, next_obs, dones) ou None.
    """
    T = len(tape)
    if T == 0 or t_idx.size == 0:
        return None

    off   = np.arange(n, dtype=np.int64)
    grid  = t_idx[:, None] + off[None, :]          # (M, n) indices absolutos
    valid = grid < T
    gclip = np.minimum(grid, T - 1)

    cur_m = tape.cur_idx[gclip]                    # (M, n)
    nxt_m = tape.next_idx[gclip]

    # transition_reward faz broadcast de (M,1) contra (M,n)
    r_m, d_m = env.transition_reward(cur_m, nxt_m, goals[:, None])
    r_m = np.where(valid, r_m, 0.0).astype(np.float32)
    d_m = d_m & valid

    any_term = d_m.any(axis=1)
    term_at  = np.argmax(d_m, axis=1)              # 0 quando nao ha terminal
    n_avail  = np.minimum(n, T - t_idx)            # passos realmente existentes
    L        = np.where(any_term, term_at + 1, n_avail)

    keep = any_term | (n_avail >= n)
    if not keep.any():
        return None

    disc = (gamma ** off).astype(np.float32)       # (n,)
    mask = off[None, :] < L[:, None]               # (M, n)
    R    = (r_m * disc[None, :] * mask).sum(axis=1).astype(np.float32)

    end = t_idx + L - 1                            # ultima transicao da janela

    # O prefixo da observacao e invariante ao goal, entao basta reescrever a
    # fatia final: obs no inicio da janela, next_obs no fim dela.
    obs      = tape.obs[t_idx].copy()
    next_obs = tape.next_obs[end].copy()
    env.write_goal_slice(obs,      tape.cur_idx[t_idx], goals)
    env.write_goal_slice(next_obs, tape.next_idx[end],  goals)

    return (obs[keep], tape.action[t_idx][keep], R[keep],
            next_obs[keep], any_term[keep])


# ─────────────────────────────────────────────────────────────────────────────
# 5. Avaliacao
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(agent: TorchDDQN, env: DynamicGoalMazeEnv, topo: MazeTopology,
             pairs: np.ndarray, max_steps: int) -> Dict[str, float]:
    """Politica deterministica (eps=0) sobre o conjunto held-out.

    Reporta:
      success_rate — % de pares resolvidos dentro de max_steps
      opt_ratio    — media de passos_tomados / passos_otimos, so nos sucessos.
                     1.0 = otimo. Sucesso sozinho nao diz se a politica e boa:
                     chegar ao goal em 10x o caminho minimo ainda e "sucesso".
      mean_steps   — passos medios nos sucessos
    """
    was_training = agent.policy_net.training
    old_eps      = agent.epsilon
    agent.policy_net.eval()
    agent.epsilon = 0.0

    successes, ratios, steps_list = 0, [], []
    for s, g in pairs:
        state = env.reset(int(s), int(g))
        opt   = topo.optimal(int(s), int(g))
        for t in range(1, max_steps + 1):
            action = agent.act(state, eval=True)
            state, _, done, _ = env.step(action)
            if done:
                successes += 1
                steps_list.append(t)
                if opt > 0:
                    ratios.append(t / opt)
                break

    agent.epsilon = old_eps
    if was_training:
        agent.policy_net.train()

    n = max(1, len(pairs))
    return {
        "success_rate": 100.0 * successes / n,
        "opt_ratio":    float(np.mean(ratios))     if ratios     else float("nan"),
        "mean_steps":   float(np.mean(steps_list)) if steps_list else float("nan"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5b. Modelos — baseline denso e ReservedSAECollab
# ─────────────────────────────────────────────────────────────────────────────

DENSE = "dense"
RSAE  = "reserved_sae"

# Defaults por modelo. Sao aplicados apenas quando a flag correspondente NAO foi
# passada na linha de comando (o argparse usa default=None para distinguir
# "nao informado" de "informado com o valor do default").
#
# Os valores de `reserved_sae` vem de ablation/experiments.py::fast_experiment_1,
# conforme pedido. Excecoes deliberadas, todas anotadas:
#   - encoder/possible_actions: ONE_HOT + sensor de paredes (pedido explicito)
#   - max_steps: derivado do pior caminho otimo, nao de 4*opens_count — no
#     regime goal-conditioned o episodio tem alvo variavel
#   - episodes 400: e o valor do fast_experiment_1, mas ver o aviso emitido em
#     train_dynamic (400 episodios e pouco para start/goal dinamicos)
MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    DENSE: {
        "encoder":          StateEncoder.COORDS_NORM.value,
        "possible_actions": True,
        "episodes":         3000,
        "lr":               1e-4,
        "gamma":            0.98,
        "batch":            128,
        "learn":            8,
        "hidden":           [256, 256],
        "min_replay_mult":  (2_000, 4),
        "eps_decay_frac":   0.5,
    },
    RSAE: {
        "encoder":             StateEncoder.ONE_HOT.value,
        "possible_actions":    True,
        "episodes":            400,
        "lr":                  1e-5,      # fast_experiment_1: learning_rate
        "new_layer_lr":        5e-5,      # fast_experiment_1: new_layer_learning_rate
        "gamma":               0.999,     # fast_experiment_1: discount_factor
        "batch":               512,       # fast_experiment_1: batch_size
        "learn":               4,         # fast_experiment_1: LEARN_STEPS
        "max_layers":          4,         # fast_experiment_1: N_MAX_LAYERS
        "insertion":           "CRT",
        "layer_mode":          "M4",
        "mutation":            "Hidden",
        "insert_patience":     15,        # fast_experiment_1: insert_patience
        "insert_min_goals":    5,         # fast_experiment_1: insert_min_goals
        "insert_min_variance": 0.6,       # fast_experiment_1: insert_min_variance
        "rolling_window":      20,        # fast_experiment_1: rolling_window_size
        "min_replay_mult":     (1_000, 2),
        "eps_decay_frac":      1.0,
    },
}


def resolve_defaults(model: str, ns: argparse.Namespace) -> None:
    """Preenche in-place os campos None de `ns` com o default do modelo.

    Assim `--lr` passado na CLI sempre vence, e nao informar `--lr` da o valor
    certo para o modelo escolhido em vez de um default unico e errado para um
    dos dois.
    """
    d = MODEL_DEFAULTS[model]
    for key, val in d.items():
        if key in ("min_replay_mult", "eps_decay_frac"):
            continue
        if getattr(ns, key, None) is None:
            setattr(ns, key, val)


def default_model_arch(max_layers: int,
                       sae_hidden: Optional[int] = None,
                       sae_extra: Optional[int] = None,
                       width_delta: Optional[float] = None) -> ModelArch:
    """ModelArch do SAE. Dois regimes, os mesmos que existem em ablation/:

    PROPORCIONAL (default, = ARCHITECTURES[0] do fast_experiment_1)
        sae_hidden None -> os multiplicadores sao FRACOES de base_width, e
        base_width = action_size * rows * cols (o tamanho de uma Q-table
        tabular). hidden = 1/2 * base_width, extra = 1/2 * base_width.
        As larguras acompanham o tamanho do labirinto.

    ESTATICO (= fast_experiment_start_equal, is_static=True)
        sae_hidden informado -> gen_concrete_arch forca base_width = 1, entao
        os "multiplicadores" viram contagens ABSOLUTAS de neuronios. E o que
        permite fixar a largura independentemente do maze.

    Em ambos, `width_delta` e a fracao somada por camada (CRT/DRT/ALT).
    """
    wd = width_delta if width_delta is not None else 1.0 / max_layers
    activation = LayersConfig(nn.ReLU, nn.Identity, nn.ReLU)
    use_bias   = LayersConfig(True, True, True)

    if sae_hidden is None:
        return ModelArch(
            max_layers,
            LayersConfig(1 / 2, 1, 1 / 2),
            LayersConfig(wd, 1, wd),
            activation, use_bias,
        )

    extra = sae_extra if sae_extra is not None else sae_hidden
    return ModelArch(
        max_layers,
        LayersConfig(int(sae_hidden), 1, int(extra)),
        LayersConfig(wd, 1, wd),
        activation, use_bias,
        is_static=True,
    )


def build_reserved_layers(env: DynamicGoalMazeEnv,
                          concrete_arch: List[LayersConfig],
                          model_arch: ModelArch,
                          mode_type: LayerModeType,
                          mutation_mode: MutationMode,
                          eta_increment: float) -> List[NewLayerCfg]:
    """Mesma montagem de ablation/training.py::train_reserved_saecollab_tolerance_model.

    A primeira camada nasce ativa e sem mutacao; as demais nascem congeladas e
    sao liberadas uma a uma por use_next_layer().
    """
    cfgs = [NewLayerCfg(
        hidden_dim        = int(concrete_arch[0].hidden),
        out_dim           = env.action_size,
        extra_dim         = None,
        mutation_mode     = None,
        target_fn         = None,
        k                 = 1.0,
        eta               = 0.0,
        eta_increment     = eta_increment,
        hidden_activation = model_arch.activation.hidden(),
        out_activation    = model_arch.activation.out(),
        extra_activation  = model_arch.activation.extra(),
        is_k_trainable    = mode_type.value.is_k_trainable,
        use_bias          = model_arch.use_bias,
    )]

    for i in range(1, model_arch.max_layers):
        extra_dim = int(concrete_arch[i].extra) if mode_type.value.use_extra_branch else None
        extra_dim = None if extra_dim == 0 else extra_dim
        cfgs.append(NewLayerCfg(
            hidden_dim        = int(concrete_arch[i].hidden),
            out_dim           = env.action_size,
            extra_dim         = extra_dim,
            mutation_mode     = mutation_mode,
            target_fn         = model_arch.activation.hidden(),
            k                 = 1.0,
            eta               = 0.0,
            eta_increment     = eta_increment,
            hidden_activation = model_arch.activation.hidden(),
            out_activation    = model_arch.activation.out(),
            extra_activation  = model_arch.activation.extra(),
            is_k_trainable    = mode_type.value.is_k_trainable,
            use_bias          = model_arch.use_bias,
        ))
    return cfgs


def sae_param_count(agent: ReservedSAECollabDDQN) -> int:
    """So as camadas ativas contam — as reservadas ainda estao congeladas."""
    return sum(
        p.numel()
        for layer in agent.policy_net.layers[:agent.policy_net.active_head + 1]
        for p in layer.parameters()
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5c. Recarregar um treino + avaliacao de cobertura total
# ─────────────────────────────────────────────────────────────────────────────

def q_values(policy_net, x: torch.Tensor) -> torch.Tensor:
    """Q(s,.) de qualquer um dos dois agentes.

    TorchDDQN devolve um tensor; ReservedSAECollabNet devolve
    (q_values, outputs, hiddens).
    """
    out = policy_net(x)
    return out[0] if isinstance(out, tuple) else out


def load_trained_run(run_dir: str, model_name: str = "model_last.pth",
                     maze_override: Optional[str] = None,
                     device: str = "cpu", verbose: bool = False):
    """Reconstroi (maze, topo, env, agent, summary, model_path) de um treino.

    Levanta excecao em vez de sys.exit para poder ser usada como biblioteca —
    o viewer e o script de comparacao dependem os dois desta funcao, e
    duplica-la seria garantir que um dia divergissem.
    """
    summary_path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary.json nao encontrado em {run_dir}")

    with open(summary_path) as f:
        s = json.load(f)

    maze_path = maze_override or s.get("maze_path") or f"./mazes/{s['maze']}.maze"
    if not os.path.exists(maze_path):
        raise FileNotFoundError(f"maze nao encontrado: {maze_path}")

    model_path = os.path.join(run_dir, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"modelo nao encontrado: {model_path}")

    maze = MazeEnv(maze_path, rewards_scaled=False, pass_through_walls=False)
    if not maze.file_loaded:
        raise RuntimeError(f"falha lendo {maze_path}")

    topo = MazeTopology(maze, verbose=verbose)
    env  = DynamicGoalMazeEnv(
        maze, topo,
        state_encoder            = StateEncoder(s.get("encoder", "coords_norm")),
        num_last_states          = s.get("num_last_states", 0),
        num_last_actions         = s.get("num_last_actions", 0),
        possible_actions_feature = s.get("possible_actions", True),
        use_shaping              = s.get("use_shaping", True),
        shaping_gamma            = s.get("gamma", 0.98),
        device                   = device,
    )

    expected = s.get("state_size")
    if expected is not None and expected != env.state_size:
        raise RuntimeError(f"state_size do treino ({expected}) != reconstruido "
                           f"({env.state_size})")

    if s.get("model", DENSE) == RSAE:
        # ReservedSAECollabDDQN.save() serializa o MODULO inteiro; o load()
        # substitui policy_net. A cfg abaixo so precisa deixar o construtor
        # rodar, mas e montada com os valores reais do summary para que
        # divergencia vire erro, e nao comportamento estranho.
        max_layers  = s.get("max_layers", 4)
        model_arch  = default_model_arch(max_layers, s.get("sae_hidden"),
                                         s.get("sae_extra"),
                                         s.get("sae_width_delta"))
        insert_type = LayerInsertionType(s.get("insertion", "CRT"))
        mode_type   = LayerModeType.from_tag(s.get("layer_mode", "M4"))
        mut_mode    = MutationMode[s.get("mutation", "Hidden")]
        base_width  = env.action_size * topo.rows * topo.cols
        concrete    = gen_concrete_arch(base_width, env, model_arch, insert_type)
        reserved    = build_reserved_layers(
            env, concrete, model_arch, mode_type, mut_mode,
            eta_increment=1.0 / max(1, s.get("episodes", 400)),
        )
        agent = ReservedSAECollabDDQN(
            state_size          = env.state_size,
            action_size         = env.action_size,
            reserved_layers_cfg = reserved,
            lr                  = [s.get("lr", 1e-5), s.get("new_layer_lr", 5e-5)],
            use_bias            = model_arch.use_bias,
            device              = device,
        )
    else:
        agent = TorchDDQN(
            sequential_list = list(s.get("hidden", [256, 256])),
            state_size      = env.state_size,
            action_size     = env.action_size,
            device          = device,
        )

    agent.load(model_path)
    agent.epsilon = 0.0
    agent.policy_net.to(agent.device)
    agent.policy_net.eval()
    return maze, topo, env, agent, s, model_path


def all_valid_pairs(topo: MazeTopology) -> np.ndarray:
    """Todos os pares (start, goal) navegaveis, distintos e alcancaveis."""
    nav    = topo.nav
    sub    = topo.dist[np.ix_(nav, nav)]
    ii, jj = np.nonzero(sub > 0)
    return np.stack([nav[ii], nav[jj]], axis=1).astype(np.int32)


@torch.no_grad()
def full_coverage_eval(agent, env: DynamicGoalMazeEnv, topo: MazeTopology,
                       max_steps: int,
                       max_pairs: Optional[int] = None,
                       seed: int = 0) -> Dict[str, Any]:
    """Avalia em TODOS os pares validos (ou numa amostra deterministica).

    E a medida honesta: o held-out de 256 pares ja demonstrou inverter a ordem
    de dois checkpoints. `by_distance` mostra ate onde a politica alcanca, que
    e o eixo em que estes agentes falham.
    """
    pairs = all_valid_pairs(topo)
    sampled = False
    if max_pairs is not None and len(pairs) > max_pairs:
        idx     = np.random.default_rng(seed).choice(len(pairs), max_pairs,
                                                     replace=False)
        pairs   = pairs[np.sort(idx)]
        sampled = True

    agent.policy_net.eval()
    old_eps, agent.epsilon = agent.epsilon, 0.0

    ok_d: Dict[int, int] = {}
    tot_d: Dict[int, int] = {}
    ratios: List[float] = []
    successes = 0

    for s_i, g_i in pairs:
        s_i, g_i = int(s_i), int(g_i)
        d = topo.optimal(s_i, g_i)
        tot_d[d] = tot_d.get(d, 0) + 1
        state = env.reset(s_i, g_i)
        for t in range(1, max_steps + 1):
            state, _, done, _ = env.step(agent.act(state, eval=True))
            if done:
                successes += 1
                ok_d[d] = ok_d.get(d, 0) + 1
                ratios.append(t / d)
                break

    agent.epsilon = old_eps
    n = max(1, len(pairs))
    return {
        "n_pairs":      int(len(pairs)),
        "sampled":      sampled,
        "success_rate": 100.0 * successes / n,
        "opt_ratio":    float(np.mean(ratios)) if ratios else float("nan"),
        "by_distance":  {int(d): (ok_d.get(d, 0), tot_d[d]) for d in sorted(tot_d)},
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Treino
# ─────────────────────────────────────────────────────────────────────────────

def train_dynamic(
    maze_path: str,
    out_dir: str,
    model: str               = DENSE,
    seed: int                = 333,
    episodes: int            = 3000,
    max_steps: Optional[int] = None,
    encoder: StateEncoder    = StateEncoder.COORDS_NORM,
    possible_actions: bool   = True,
    num_last_states: int     = 0,
    num_last_actions: int    = 0,
    hidden: Sequence[int]    = (256, 256),
    # ── ReservedSAECollab ────────────────────────────────────────────────
    max_layers: int          = 4,
    sae_hidden: Optional[int]      = None,
    sae_extra: Optional[int]       = None,
    sae_width_delta: Optional[float] = None,
    insertion: str           = "CRT",
    layer_mode: str          = "M4",
    mutation: str            = "Hidden",
    new_layer_lr: float      = 5e-5,
    insert_criterion: str    = "plateau",
    insert_patience: int     = 15,
    insert_min_goals: int    = 5,
    insert_min_variance: float = 0.6,
    insert_patience_evals: int = 2,
    insert_min_delta: float  = 1.0,
    insert_skip_success: float = 95.0,
    # ── criterio 'slope' (ver bloco de insercao) ─────────────────────────
    insert_warmup_frac: float  = 0.25,
    insert_tail_frac: float    = 0.20,
    insert_regress_guard: float = 5.0,
    insert_pace: bool          = True,
    rolling_window: int      = 20,
    # ─────────────────────────────────────────────────────────────────────
    lr: float                = 1e-4,
    gamma: float             = 0.98,
    batch_size: int          = 128,
    learn_interval: int      = 8,
    use_her: bool            = True,
    her_k: int               = 4,
    n_step: int              = 3,
    use_shaping: bool        = True,
    # 64 pares e pequeno demais: 1 par vale 1.6pp e as oscilacoes entre
    # avaliacoes afogam o sinal. Medido no small_eg, o eval de 64 pares elegeu
    # como model_best um checkpoint 5.8pp PIOR na cobertura total (2862 pares)
    # do que o descartado. Com 256 o erro de estimativa cai para ~3pp.
    eps_decay_frac: Optional[float] = None,
    min_replay: Optional[int]       = None,
    eval_pairs: int          = 256,
    eval_interval: int       = 50,
    curriculum: bool         = False,
    device: Optional[str]    = None,
    verbose: bool            = True,
) -> Dict[str, Any]:

    set_seed(seed)
    rng      = np.random.default_rng(seed)
    eval_rng = np.random.default_rng(seed + 10_000)   # independente do treino
    device   = device or ("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(out_dir, exist_ok=True)

    maze = MazeEnv(maze_path, rewards_scaled=False, pass_through_walls=False)
    if not maze.file_loaded:
        raise FileNotFoundError(maze_path)

    topo = MazeTopology(maze, verbose=verbose)
    env  = DynamicGoalMazeEnv(
        maze, topo,
        state_encoder            = encoder,
        num_last_states          = num_last_states,
        num_last_actions         = num_last_actions,
        possible_actions_feature = possible_actions,
        use_shaping              = use_shaping,
        shaping_gamma            = gamma,
        device                   = device,
    )

    if max_steps is None:
        # Teto generoso: o pior caminho otimo do maze, com folga para exploracao.
        max_steps = int(min(4 * topo.worst_optimal(), 4 * len(topo.nav)))
    max_steps = max(int(max_steps), 8)

    # Held-out fixo; o treino e proibido de amostra-lo.
    eval_set = make_eval_set(topo, eval_rng, eval_pairs)
    excluded = {int(s) * topo.n + int(g) for s, g in eval_set}
    sampler  = PairSampler(topo, rng, exclude=excluded)

    # HER multiplica as transicoes empurradas por episodio; sem compensar o
    # learn_interval o numero de gradient steps deixa de ser comparavel ao
    # baseline sem HER.
    effective_interval = learn_interval * (1 + her_k) if use_her else learn_interval

    # dense: eps chega em 0.1 na metade do treino. reserved_sae: no fim, como
    # em fast_experiment_1 (final_step = MAX_STEPS * EPISODES).
    # ATENCAO ao comparar modelos: este valor difere entre eles por default, e
    # muda MUITO quanta exploracao cada um faz (ao fim de 3000 episodios, 0.5 da
    # eps~0.11 e 1.0 da eps~0.31). Iguale com --eps-decay-frac.
    eps_frac = (eps_decay_frac if eps_decay_frac is not None
                else MODEL_DEFAULTS[model]["eps_decay_frac"])
    epsilon_decay = exp_decay_factor_to(
        final_epsilon=0.1, final_step=max_steps * episodes * eps_frac,
        epsilon_start=1.0, convergence_threshold=0.01,
    )

    n_step = max(1, int(n_step))

    # O agente recebe gamma ** n_step, NAO gamma. O learn() de ambos os agentes
    # faz  target = R + self.gamma * Q(next) * (1 - done)  e o alvo n-step
    # correto e  target = R + gamma^n * Q(s_{t+n}) * (1 - done), com R ja
    # descontado internamente por gamma (feito em build_nstep). Nas janelas que
    # terminam em goal, (1 - done) zera o bootstrap e o expoente e irrelevante.
    agent_gamma = gamma ** n_step
    floor_, mult_ = MODEL_DEFAULTS[model]["min_replay_mult"]
    min_replay    = (int(min_replay) if min_replay is not None
                     else max(floor_, mult_ * batch_size))

    is_sae        = (model == RSAE)
    concrete_arch = None
    model_arch    = None

    if not is_sae:
        agent = TorchDDQN(
            sequential_list = list(hidden),
            state_size      = env.state_size,
            action_size     = env.action_size,
            lr              = lr,
            gamma           = agent_gamma,
            batch_size      = batch_size,
            buffer_size     = 400_000,
            epsilon_start   = 1.0,
            epsilon_final   = 0.1,
            epsilon_decay   = epsilon_decay,
            learn_interval  = effective_interval,
            min_replay_size = min_replay,
            device          = device,
        )
    else:
        mode_type     = LayerModeType.from_tag(layer_mode)
        mutation_mode = MutationMode[mutation]
        insert_type   = LayerInsertionType(insertion)
        model_arch    = default_model_arch(max_layers, sae_hidden, sae_extra,
                                           sae_width_delta)

        # Mesma largura base do fast_experiment_1: o tamanho de uma Q-table
        # tabular (acoes * celulas), que gen_concrete_arch depois escala pelos
        # multiplicadores da ModelArch.
        base_width    = env.action_size * topo.rows * topo.cols
        concrete_arch = gen_concrete_arch(base_width, env, model_arch, insert_type)

        reserved_cfg = build_reserved_layers(
            env, concrete_arch, model_arch, mode_type, mutation_mode,
            eta_increment = 1.0 / max(1, episodes),
        )
        agent = ReservedSAECollabDDQN(
            state_size          = env.state_size,
            action_size         = env.action_size,
            reserved_layers_cfg = reserved_cfg,
            accelerate_etas     = True,
            accelerate_factor   = 2.0,
            lr                  = [lr, new_layer_lr],
            gamma               = agent_gamma,
            batch_size          = batch_size,
            buffer_size         = 400_000,
            epsilon_start       = 1.0,
            epsilon_final       = 0.1,
            epsilon_decay       = epsilon_decay,
            learn_interval      = effective_interval,
            min_replay_size     = min_replay,
            use_bias            = model_arch.use_bias,
            device              = device,
        )

    if verbose:
        print("=" * 72)
        print(f"  modelo        : {model}")
        print(f"  maze          : {maze.tag} ({topo.rows}x{topo.cols}), "
              f"{topo.nav.size} celulas navegaveis")
        print(f"  pares validos : {topo.n_valid_pairs():,}  "
              f"(held-out: {len(eval_set)})")
        print(f"  encoder       : {encoder.value}  ->  state_size = {env.state_size}")
        print(f"  goal slice    : [{env.goal_off}:{env.state_size}] "
              f"(enc {env.base_enc_size} + delta 2)")
        print(f"  max_steps     : {max_steps}   episodes: {episodes}")
        print(f"  HER           : {'future k=' + str(her_k) if use_her else 'OFF'}"
              f"   shaping: {'ON' if use_shaping else 'OFF'}")
        print(f"  n-step        : {n_step}  (gamma {gamma} -> gamma_agente "
              f"{gamma ** n_step:.5f})")
        print(f"  learn_interval: {effective_interval} (base {learn_interval})")
        if is_sae:
            widths = [int(c.hidden) for c in concrete_arch]
            extras = [int(c.extra)  for c in concrete_arch]
            print(f"  SAE           : {max_layers} camadas  {insertion} / "
                  f"{layer_mode} / {mutation}")
            if model_arch.is_static:
                print(f"  larguras de   : --sae-hidden {sae_hidden} "
                      f"(estatico, independente do maze)")
            else:
                print(f"  larguras de   : 1/2 x base_width, base_width = "
                      f"{env.action_size}x{topo.rows}x{topo.cols} = {base_width} "
                      f"(proporcional ao maze; use --sae-hidden para fixar)")
            print(f"  hidden        : {widths}")
            print(f"  extra         : {extras}"
                  f"{'' if mode_type.value.use_extra_branch else '  (ignorado: modo sem extra branch)'}")
            if insert_criterion == "slope":
                W = insert_patience_evals
                print(f"  insercao      : slope - media das ultimas {W} avaliacoes "
                      f"nao supera a das {W} anteriores em {insert_min_delta:.1f}pp")
                print(f"                  janela viva: ep "
                      f"{int(insert_warmup_frac * episodes)} a "
                      f"{int((1 - insert_tail_frac) * episodes)} "
                      f"(warmup {insert_warmup_frac:.0%}, cauda reservada "
                      f"{insert_tail_frac:.0%})")
                print(f"                  guarda de regressao: nao insere a mais de "
                      f"{insert_regress_guard:.1f}pp abaixo do melhor")
                if insert_pace:
                    _lo = insert_warmup_frac * episodes
                    _hi = (1 - insert_tail_frac) * episodes
                    _sl = (_hi - _lo) / max(1, max_layers - 1)
                    _dl = [int(_lo + (k + 1) * _sl) for k in range(max_layers - 1)]
                    print(f"                  prazos por fatia: {_dl} "
                          f"(escape: insere mesmo sem estagnacao)")
            elif insert_criterion == "plateau":
                print(f"  insercao      : plateau - {insert_patience_evals} avaliacoes "
                      f"sem ganhar {insert_min_delta:.1f}pp no held-out")
                print(f"                  (cadencia amarrada a --eval-interval "
                      f"{eval_interval}: 1a insercao possivel no ep "
                      f"{eval_interval * (insert_patience_evals + 1)})")
            else:
                print(f"  insercao      : variance - patience {insert_patience}, "
                      f"min_goals {insert_min_goals}, min_var {insert_min_variance}")
            print(f"  skip se       : held-out >= {insert_skip_success:.0f}%")
            print(f"  lr            : {lr} (camada nova: {new_layer_lr})")
        if not is_sae:
            print(f"  hidden        : {list(hidden)}")
        print(f"  curriculum    : {'ON (max_dist em rampa)' if curriculum else 'OFF'}")
        print(f"  device        : {device}")
        print("=" * 72)
        if is_sae and episodes < 1000:
            print(f"[AVISO] episodes={episodes} vem do fast_experiment_1, que treina "
                  f"UM par (start,goal) fixo.")
            print(f"        Aqui o alvo e variavel entre {topo.n_valid_pairs():,} pares; "
                  f"o baseline denso so passou de 60% depois de ~2000 episodios.")
            print(f"        Considere --episodes 3000 para comparar de igual para igual.")
            print("=" * 72)

    tape      = EpisodeTape(max_steps, env.state_size)
    rows: List[Dict[str, Any]] = []
    best_eval = -1.0
    t_start   = time.perf_counter()
    worst_opt = topo.worst_optimal()

    # ── estado da insercao de camadas (so usado quando is_sae) ──────────────
    current_branch    = 0
    max_branches      = (max_layers - 1) if is_sae else 0   # a 1a ja nasce ativa
    eps_since_branch  = 0
    goal_once_reached = False
    last_eval_success = 0.0
    ep_rewards: List[float] = []
    ep_reached: List[int]   = []
    # criterio 'plateau': historico das avaliacoes no held-out e o indice a
    # partir do qual ele conta (reiniciado a cada camada, para que a camada
    # nova seja julgada pelo progresso DELA, nao pelo acumulado antes).
    eval_history: List[float] = []
    evals_at_last_branch      = 0

    for episode in range(episodes):
        ep_t0 = time.perf_counter()

        if curriculum:
            # Rampa simples por distancia BFS; so ativa com --curriculum.
            frac = min(1.0, (episode + 1) / (0.7 * episodes))
            sampler.max_dist = int(max(2, frac * worst_opt))

        start_idx, goal_idx = sampler.sample()
        state = env.reset(start_idx, goal_idx)
        tape.reset()

        cum_reward = 0.0
        reached    = False
        for step in range(max_steps):
            action = agent.act(state, eval=False)
            prev   = env._obs.copy()          # obs ANTES do step (com goal real)
            next_state, reward, done, info = env.step(action)

            tape.add(prev, action, reward, env._obs, done,
                     info["cur_idx"], info["next_idx"])

            cum_reward += reward
            state = next_state
            if done:
                reached = True
                break

        T = len(tape)

        def push(batch) -> int:
            if batch is None:
                return 0
            b_obs, b_act, b_ret, b_next, b_done = batch
            for i in range(len(b_act)):
                agent.remember(b_obs[i], int(b_act[i]), float(b_ret[i]),
                               b_next[i], bool(b_done[i]))
            return len(b_act)

        # ── transicoes reais (janelas n-step sob o goal do episodio) ────────
        n_real = 0
        if T > 0:
            t_all  = np.arange(T, dtype=np.int64)
            g_all  = np.full(T, goal_idx, dtype=np.int64)
            n_real = push(build_nstep(env, tape, t_all, g_all, n_step, gamma))

        # ── transicoes relabeladas pelo HER ─────────────────────────────────
        n_her = 0
        if use_her and T > 0:
            sampled = sample_future_goals(tape, her_k, rng)
            if sampled is not None:
                h_t, h_g = sampled
                n_her = push(build_nstep(env, tape, h_t, h_g, n_step, gamma))

        if is_sae:
            agent.policy_net.step_all_etas()
        agent.update_epsilon()

        ep_rewards.append(cum_reward)
        ep_reached.append(int(reached))
        goal_once_reached = goal_once_reached or reached

        # ── insercao de camada, criterio 'variance' ─────────────────────────
        # Criterio de tolerancia original do ablation/training.py, adaptado ao
        # regime goal-conditioned em dois pontos:
        #   1. `eval_agent_deterministic` testava UM par fixo. Aqui o
        #      equivalente e a taxa de sucesso no held-out: se ja esta alta,
        #      nao ha por que gastar capacidade nova.
        #   2. `goals_in_window` somava `cumulative_goals` (contadores
        #      acumulados). Aqui conta sucessos POR episodio na janela, que e
        #      o que a comparacao com insert_min_goals pressupoe.
        #
        # AVISO MEDIDO: com start/goal dinamicos este criterio praticamente nao
        # dispara. A variancia do reward por episodio e dominada pelo SORTEIO do
        # par (distancia otima de 1 a 19 no small_eg => reward de -39 a +1), nao
        # pelo aprendizado; o var_ratio fica em 2-11 contra um limiar de 0.6, e
        # ainda SOBE conforme o agente melhora, porque o denominador |mean| cai.
        # Mantido para reproduzir o comportamento do fast_experiment_1 e para
        # comparacao. O default e 'plateau'.
        eps_since_branch += 1
        if (is_sae and insert_criterion == "variance"
                and current_branch < max_branches
                and episode >= insert_patience
                and episode % insert_patience == 0):

            if last_eval_success >= insert_skip_success:
                pass                      # ja resolve bem; nao adiciona camada
            else:
                w_r   = ep_rewards[-insert_patience:]
                w_g   = sum(ep_reached[-insert_patience:])
                w_m   = float(np.mean(w_r))
                w_v   = float(np.var(w_r))
                ratio = (w_v / abs(w_m)) if abs(w_m) > 1e-6 else float("inf")

                should_advance = (
                    ratio < insert_min_variance
                    and (w_g >= insert_min_goals
                         or not goal_once_reached
                         or current_branch == 0)
                    and eps_since_branch >= insert_patience
                )
                if should_advance:
                    agent.use_next_layer()
                    current_branch  += 1
                    eps_since_branch = 0
                    if verbose:
                        print(f"[ep {episode+1:>6}] camada {current_branch+1}/"
                              f"{max_layers} ativada  "
                              f"(var_ratio {ratio:.3f}, goals {w_g}/{insert_patience}, "
                              f"params {sae_param_count(agent):,})")

        row = {
            "episode":        episode,
            "reward":         cum_reward,
            "steps":          T,
            "reached":        int(reached),
            "optimal":        topo.optimal(start_idx, goal_idx),
            "loss":           float(agent.loss),
            "epsilon":        float(agent.epsilon),
            "real_pushed":    n_real,
            "her_pushed":     n_her,
            "active_layers":  (current_branch + 1) if is_sae else len(hidden),
            "parameters":     sae_param_count(agent) if is_sae else
                              sum(p.numel() for p in agent.policy_net.parameters()),
            "replay_size":    len(agent.replay),
            "delta_time":     time.perf_counter() - ep_t0,
            "eval_success":   float("nan"),
            "eval_opt_ratio": float("nan"),
        }

        if eval_interval > 0 and (episode + 1) % eval_interval == 0:
            ev = evaluate(agent, env, topo, eval_set, max_steps)
            row["eval_success"]   = ev["success_rate"]
            row["eval_opt_ratio"] = ev["opt_ratio"]
            last_eval_success     = ev["success_rate"]
            if ev["success_rate"] > best_eval:
                best_eval = ev["success_rate"]
                agent.save(os.path.join(out_dir, "model_best.pth"))
            if verbose:
                elapsed = time.perf_counter() - t_start
                print(f"[ep {episode+1:>6}/{episodes}] "
                      f"eval {ev['success_rate']:5.1f}%  "
                      f"opt_ratio {ev['opt_ratio']:.2f}  "
                      f"| eps {agent.epsilon:.3f}  loss {agent.loss:.4f}  "
                      f"replay {len(agent.replay):,}  "
                      f"| {elapsed:6.1f}s")

            # ── insercao de camada, criterio 'plateau' ──────────────────────
            # Mede aprendizado diretamente: se a MELHOR taxa no held-out das
            # ultimas `insert_patience_evals` avaliacoes nao superou a melhor
            # anterior por pelo menos `insert_min_delta` pontos, o modelo
            # empacou e ganha capacidade.
            #
            # Duas escolhas que evitam falso positivo:
            #   - compara maximos, nao a ultima leitura: uma queda pontual por
            #     ruido de avaliacao nao conta como platô.
            #   - `evals_at_last_branch` reinicia a janela apos cada insercao,
            #     dando a camada nova `insert_patience_evals` avaliacoes antes
            #     de ser julgada. Sem isso as camadas entrariam em rajada.
            eval_history.append(ev["success_rate"])
            n_since = len(eval_history) - evals_at_last_branch

            if (is_sae and insert_criterion == "plateau"
                    and current_branch < max_branches
                    and n_since > insert_patience_evals
                    and last_eval_success < insert_skip_success):

                window      = eval_history[-insert_patience_evals:]
                before      = eval_history[evals_at_last_branch:-insert_patience_evals]
                best_before = max(before)
                gain        = max(window) - best_before

                if gain < insert_min_delta:
                    agent.use_next_layer()
                    current_branch      += 1
                    eps_since_branch     = 0
                    evals_at_last_branch = len(eval_history)
                    if verbose:
                        print(f"[ep {episode+1:>6}] camada {current_branch+1}/"
                              f"{max_layers} ativada  "
                              f"(plateau: ganho {gain:+.1f}pp em "
                              f"{insert_patience_evals} avaliacoes, "
                              f"limiar {insert_min_delta:.1f}pp, "
                              f"params {sae_param_count(agent):,})")

            # ── insercao de camada, criterio 'slope' ───────────────────────
            #
            # Corrige tres falhas medidas no criterio 'plateau' (ver
            # dynamic_results/cmp: insercoes em ep 250-550 com held-out em
            # 15-25%, orcamento inteiro gasto ate ~ep 1500, queda de 8-14pp
            # entre o melhor checkpoint e o final):
            #
            #   (a) 'plateau' compara max(ultimas W) contra o max CORRENTE
            #       desde a ultima insercao. Um unico eval ruim no meio da
            #       subida ja zera o ganho aparente e dispara a insercao. Com
            #       W=2 isso e quase inevitavel. Aqui a comparacao e entre
            #       MEDIAS de duas janelas DISJUNTAS de W avaliacoes: mede
            #       tendencia, nao o pior ponto.
            #
            #   (b) 'plateau' pode inserir na primeira avaliacao valida, quando
            #       a camada base mal comecou a aprender. `insert_warmup_frac`
            #       reserva o inicio do treino para a topologia minima.
            #
            #   (c) congelar tudo faltando pouco treino deixa a ultima
            #       ramificacao sem tempo de convergir - e ela e a unica
            #       treinavel dali em diante. `insert_tail_frac` reserva a
            #       cauda do treino para consolidar.
            #
            # A guarda de regressao evita congelar o estado exatamente durante
            # uma queda: se o held-out caiu muito abaixo do melhor ja visto, o
            # problema nao e falta de capacidade.
            if (is_sae and insert_criterion == "slope"
                    and current_branch < max_branches
                    and last_eval_success < insert_skip_success):

                W         = max(1, insert_patience_evals)
                n_since   = len(eval_history) - evals_at_last_branch
                ep_now    = episode + 1
                usable_lo = insert_warmup_frac * episodes
                usable_hi = (1 - insert_tail_frac) * episodes
                in_window = usable_lo <= ep_now <= usable_hi
                recovered = (best_eval - last_eval_success) <= insert_regress_guard

                # (1) gatilho por estagnacao: a tendencia achatou.
                stalled, gain = False, float("nan")
                if in_window and recovered and n_since >= 2 * W:
                    now_w  = eval_history[-W:]
                    prev_w = eval_history[-2 * W:-W]
                    gain   = (sum(now_w) / len(now_w)) - (sum(prev_w) / len(prev_w))
                    stalled = gain < insert_min_delta

                # (2) escape por prazo: sem ele o criterio se auto-sabota.
                #
                # Medido em dynamic_results/cmp_slope: so com o gatilho (1),
                # duas de tres seeds terminaram o treino com UMA camada (257k
                # parametros, 70-75% de cobertura) porque um modelo sem
                # capacidade melhora devagar mas SEM PARAR - ganhou +3 a +8pp
                # por janela ate o ep 3000. Perguntar "voce parou de melhorar?"
                # a um modelo faminto recebe sempre "nao", e ele nunca ganha a
                # capacidade que o levaria ao patamar seguinte. E o mesmo
                # impasse circular do criterio por variancia, em forma branda.
                #
                # O prazo divide a janela util em `max_branches` fatias e
                # garante que o orcamento arquitetural seja gasto dentro dela,
                # ainda sobrando a cauda para a ultima ramificacao convergir.
                # O gatilho (1) continua valendo e pode antecipar a insercao.
                overdue = False
                if insert_pace and in_window:
                    slot     = (usable_hi - usable_lo) / max(1, max_branches)
                    deadline = usable_lo + (current_branch + 1) * slot
                    overdue  = ep_now >= deadline

                if stalled or overdue:
                    agent.use_next_layer()
                    current_branch      += 1
                    eps_since_branch     = 0
                    evals_at_last_branch = len(eval_history)
                    if verbose:
                        motivo = (f"slope: {gain:+.2f}pp entre medias de {W} "
                                  f"avaliacoes, limiar {insert_min_delta:.1f}pp"
                                  if stalled else "prazo da fatia esgotado")
                        print(f"[ep {episode+1:>6}] camada {current_branch+1}/"
                              f"{max_layers} ativada  ({motivo}, "
                              f"params {sae_param_count(agent):,})")

        rows.append(row)

    agent.save(os.path.join(out_dir, "model_last.pth"))
    final = evaluate(agent, env, topo, eval_set, max_steps)

    metrics_path = os.path.join(out_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Tudo que o viewer precisa para reconstruir env + rede identicos ao treino.
    # Se faltar qualquer um destes, o load_state_dict quebra por shape mismatch.
    summary = {
        "model":             model,
        "maze":              maze.tag,
        "maze_path":         maze_path,
        "seed":              seed,
        "episodes":          episodes,
        "max_steps":         max_steps,
        "encoder":           encoder.value,
        "possible_actions":  possible_actions,
        "num_last_states":   num_last_states,
        "num_last_actions":  num_last_actions,
        "hidden":            list(hidden),
        "state_size":        env.state_size,
        "action_size":       env.action_size,
        "goal_offset":       env.goal_off,
        "use_her":           use_her,
        "her_k":             her_k,
        "n_step":            n_step,
        "gamma":             gamma,
        "gamma_agent":       gamma ** n_step,
        "use_shaping":       use_shaping,
        "curriculum":        curriculum,
        "eval_pairs":        int(len(eval_set)),
        "final_success":     final["success_rate"],
        "final_opt_ratio":   final["opt_ratio"],
        "best_eval_success": best_eval,
        "wall_time_s":       time.perf_counter() - t_start,
        # Hiperparametros EFETIVOS. Os defaults diferem por modelo, entao sem
        # registra-los aqui nao da para auditar depois se uma comparacao entre
        # dense e reserved_sae foi pareada ou nao.
        "effective": {
            "lr":              lr,
            "batch_size":      batch_size,
            "learn_interval":  effective_interval,
            "learn_base":      learn_interval,
            "eps_decay_frac":  eps_frac,
            "epsilon_decay":   epsilon_decay,
            "min_replay":      min_replay,
            "her_k":           her_k,
            "n_step":          n_step,
        },
    }
    if is_sae:
        summary.update({
            "max_layers":       max_layers,
            "sae_hidden":       sae_hidden,
            "sae_extra":        sae_extra,
            "sae_width_delta":  sae_width_delta,
            "sae_is_static":    model_arch.is_static,
            "insert_criterion": insert_criterion,
            "insert_patience_evals": insert_patience_evals,
            "insert_min_delta":      insert_min_delta,
            "insert_skip_success":   insert_skip_success,
            "insert_warmup_frac":    insert_warmup_frac,
            "insert_tail_frac":      insert_tail_frac,
            "insert_regress_guard":  insert_regress_guard,
            "insert_pace":           insert_pace,
            "eval_history":     eval_history,
            "insertion":      insertion,
            "layer_mode":     layer_mode,
            "mutation":       mutation,
            "new_layer_lr":   new_layer_lr,
            "active_layers":  current_branch + 1,
            "concrete_arch":  [{"hidden": int(c.hidden), "out": int(c.out),
                                "extra": int(c.extra)} for c in concrete_arch],
            "parameters":     sae_param_count(agent),
        })
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    if verbose:
        print("=" * 72)
        print(f"  FINAL  success {final['success_rate']:.1f}%   "
              f"opt_ratio {final['opt_ratio']:.2f}   "
              f"best {best_eval:.1f}%")
        print(f"  metrics -> {metrics_path}")
        print("=" * 72)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 7. Selftest — os invariantes sem os quais o HER e indepuravel
# ─────────────────────────────────────────────────────────────────────────────

def selftest(maze_path: str = "./mazes/small_eg.maze", verbose: bool = True) -> bool:
    print("=" * 72)
    print("  SELFTEST")
    print("=" * 72)
    rng  = np.random.default_rng(0)
    maze = MazeEnv(maze_path, pass_through_walls=False)
    assert maze.file_loaded, f"maze nao carregado: {maze_path}"
    topo = MazeTopology(maze, verbose=verbose)

    for enc in (StateEncoder.COORDS_NORM, StateEncoder.MULTI_HOT, StateEncoder.ONE_HOT):
        env = DynamicGoalMazeEnv(maze, topo, state_encoder=enc,
                                 possible_actions_feature=True, use_shaping=True,
                                 shaping_gamma=0.98)

        # ── T1: layout — a fatia do goal e contigua e termina o vetor ───────
        assert env.goal_off + env.base_enc_size + 2 == env.state_size, \
            f"[{enc.value}] goal slice nao fecha o vetor"
        assert env.goal_slice.stop == env.state_size

        # ── T2: o prefixo e invariante ao goal ──────────────────────────────
        s  = int(topo.nav[0])
        g1 = int(topo.nav[-1])
        g2 = int(topo.nav[len(topo.nav) // 2])
        o1 = env.reset(s, g1).cpu().numpy()[0].copy()
        o2 = env.reset(s, g2).cpu().numpy()[0].copy()
        assert np.array_equal(o1[:env.goal_off], o2[:env.goal_off]), \
            f"[{enc.value}] prefixo mudou ao trocar o goal -> HER quebraria"
        assert not np.array_equal(o1[env.goal_off:], o2[env.goal_off:]), \
            f"[{enc.value}] goal slice nao mudou ao trocar o goal"

        # ── T3: write_goal_slice reproduz o que o env produz ────────────────
        forged = o2.copy()
        env.write_goal_slice(forged, s, g1)
        assert np.allclose(forged, o1, atol=1e-6), \
            f"[{enc.value}] relabel nao reproduz a observacao do env"

        # ── T4: reward/done reconstruivel de (cur, next, goal) ──────────────
        for _ in range(300):
            start = int(rng.choice(topo.nav))
            goal  = int(rng.choice(topo.nav))
            if start == goal or topo.dist[start, goal] < 1:
                continue
            env.reset(start, goal)
            a = int(rng.integers(0, env.action_size))
            _, r_env, d_env, info = env.step(a)
            r_rec, d_rec = env.transition_reward(info["cur_idx"], info["next_idx"], goal)
            assert abs(float(r_rec) - r_env) < 1e-5, \
                f"[{enc.value}] reward reconstruido {float(r_rec)} != env {r_env}"
            assert bool(d_rec) == d_env, f"[{enc.value}] done reconstruido difere"

        print(f"  [OK] {enc.value:<12} state_size={env.state_size:<5} "
              f"goal_off={env.goal_off:<5} T1..T4")

    # ── T5: HER com g' = goal real reproduz a transicao original ────────────
    # E o teste mais importante: se relabelar com o proprio goal nao devolve
    # exatamente a transicao original, o relabel tem bug.
    env   = DynamicGoalMazeEnv(maze, topo, state_encoder=StateEncoder.COORDS_NORM,
                               possible_actions_feature=True, use_shaping=True,
                               shaping_gamma=0.98)
    start = int(topo.nav[0])
    goal  = int(topo.nav[-1])
    tape  = EpisodeTape(64, env.state_size)
    env.reset(start, goal)
    for _ in range(40):
        a    = int(rng.integers(0, env.action_size))
        prev = env._obs.copy()
        _, r, d, info = env.step(a)
        tape.add(prev, a, r, env._obs, d, info["cur_idx"], info["next_idx"])
        if d:
            break

    T     = len(tape)
    GAMMA = 0.98
    t_all = np.arange(T, dtype=np.int64)
    same  = np.full(T, goal, dtype=np.int64)

    # n=1 sob o goal real tem que devolver a fita original, bit a bit.
    b_obs, b_act, b_ret, b_next, b_done = build_nstep(env, tape, t_all, same, 1, GAMMA)
    assert np.allclose(b_obs,  tape.obs[:T],      atol=1e-6), "T5: obs difere"
    assert np.allclose(b_next, tape.next_obs[:T], atol=1e-6), "T5: next_obs difere"
    assert np.allclose(b_ret,  tape.reward[:T],   atol=1e-5), "T5: reward difere"
    assert np.array_equal(b_done, tape.done[:T]),             "T5: done difere"
    assert np.array_equal(b_act,  tape.action[:T]),           "T5: action difere"
    print(f"  [OK] n=1 identity  build_nstep(n=1, g_real) == fita original  (T={T})")

    # ── T9: retorno n-step bate com a soma descontada calculada na mao ──────
    for n in (2, 3, 5):
        r_obs, r_act, r_ret, r_next, r_done = build_nstep(env, tape, t_all, same, n, GAMMA)
        # recomputa janela por janela, em Python puro, sem vetorizacao
        kept, expect = [], []
        for t in range(T):
            acc, L, term = 0.0, 0, False
            for i in range(n):
                if t + i >= T:
                    break
                rr, dd = env.transition_reward(int(tape.cur_idx[t + i]),
                                               int(tape.next_idx[t + i]), goal)
                acc += (GAMMA ** i) * float(rr)
                L   += 1
                if bool(dd):
                    term = True
                    break
            if term or (T - t) >= n:      # mesma regra de descarte
                kept.append(t)
                expect.append((acc, term, t + L - 1))
        assert len(kept) == len(r_ret), \
            f"T9(n={n}): {len(r_ret)} janelas mantidas, esperado {len(kept)}"
        for k_i, (acc, term, end) in enumerate(expect):
            assert abs(r_ret[k_i] - acc) < 1e-4, \
                f"T9(n={n}) janela {kept[k_i]}: retorno {r_ret[k_i]} != {acc}"
            assert bool(r_done[k_i]) == term, f"T9(n={n}) janela {kept[k_i]}: done difere"
            assert np.allclose(r_next[k_i], tape.next_obs[end], atol=1e-6), \
                f"T9(n={n}) janela {kept[k_i]}: next_obs nao e o fim da janela"
        print(f"  [OK] n-step n={n}    {len(r_ret):>3} janelas conferidas contra "
              f"soma descontada manual")

    # ── T6: 'future' precisa gerar sucessos sinteticos ──────────────────────
    sampled = sample_future_goals(tape, 4, rng)
    assert sampled is not None, "T6: sample_future_goals devolveu None"
    h_t, h_g = sampled
    out = build_nstep(env, tape, h_t, h_g, 3, GAMMA)
    assert out is not None, "T6: build_nstep devolveu None para os goals do HER"
    _, _, _, _, h_done = out
    assert h_done.any(), "T6: nenhum relabel gerou done=True -> HER inutil"
    print(f"  [OK] HER future    {len(h_done)} janelas n-step, "
          f"{int(h_done.sum())} sucessos sinteticos "
          f"({100 * h_done.mean():.1f}%)")

    # ── T10: sob goal ficticio, a janela nunca ultrapassa a terminacao ──────
    # Se o agente pisa em g' no meio da janela, o retorno tem que parar ali.
    bad = 0
    for t_w, g_w in zip(h_t[:200], h_g[:200]):
        one = build_nstep(env, tape, np.array([t_w]), np.array([g_w]), 5, GAMMA)
        if one is None:
            continue
        _, _, _, _, dn = one
        if not bool(dn[0]):
            continue
        # terminou: nenhuma transicao ANTES do fim pode ja ter atingido g'
        first = next(i for i in range(5)
                     if t_w + i < T and int(tape.next_idx[t_w + i]) == int(g_w))
        acc = 0.0
        for i in range(first + 1):
            rr, _ = env.transition_reward(int(tape.cur_idx[t_w + i]),
                                          int(tape.next_idx[t_w + i]), int(g_w))
            acc += (GAMMA ** i) * float(rr)
        chk = build_nstep(env, tape, np.array([t_w]), np.array([g_w]), 5, GAMMA)
        if abs(float(chk[2][0]) - acc) > 1e-4:
            bad += 1
    assert bad == 0, f"T10: {bad} janelas somaram passos apos a terminacao sob g'"
    print("  [OK] n-step trunca a janela na primeira terminacao sob o goal ficticio")

    # ── T7: BFS coerente (>= manhattan, paridade do grid 4-conexo) ──────────
    checked = 0
    for _ in range(200):
        s = int(rng.choice(topo.nav))
        g = int(rng.choice(topo.nav))
        d = int(topo.dist[s, g])
        if d < 1:
            continue
        sr, sc = topo.rc(s)
        gr, gc = topo.rc(g)
        man = abs(sr - gr) + abs(sc - gc)
        assert d >= man, f"T7: BFS {d} < manhattan {man}"
        assert (d - man) % 2 == 0, f"T7: paridade invalida d={d} man={man}"
        checked += 1
    print(f"  [OK] BFS           {checked} pares validados (>= manhattan, paridade)")

    # ── T8: o sampler nunca devolve par impossivel ──────────────────────────
    sampler = PairSampler(topo, rng)
    for _ in range(500):
        s, g = sampler.sample()
        assert s != g and topo.dist[s, g] >= 1, "T8: par invalido amostrado"
    print("  [OK] PairSampler   500 pares alcancaveis e distintos")

    print("=" * 72)
    print("  TODOS OS INVARIANTES PASSARAM")
    print("=" * 72)
    return True

# ─────────────────────────────────────────────────────────────────────────────
# 8. CLI (argparse)
# ─────────────────────────────────────────────────────────────────────────────

_ENCODERS = {e.value: e for e in StateEncoder}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dynamic_maze_start_goal.py",
        description="DDQN goal-conditioned (UVFA + HER + n-step) para "
                    "start/goal dinamicos em labirintos.",
        epilog="Flags nao informadas assumem o default DO MODELO escolhido "
               "(ver MODEL_DEFAULTS). Rode --show-defaults para inspeciona-los.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g = p.add_argument_group("geral")
    g.add_argument("--selftest", action="store_true",
                   help="roda os invariantes do env/HER/n-step e sai")
    g.add_argument("--show-defaults", action="store_true",
                   help="imprime os defaults de cada modelo e sai")
    g.add_argument("--model", choices=[DENSE, RSAE], default=DENSE,
                   help="arquitetura do agente (default: %(default)s)")
    g.add_argument("-m", "--maze", default="./mazes/small_eg.maze")
    g.add_argument("-s", "--seed", type=int, default=333)
    g.add_argument("-d", "--dir", dest="out_dir", default=None,
                   help="default: dynamic_results/<modelo>/<maze>/<seed>")
    g.add_argument("--device", choices=["cpu", "cuda"], default=None)
    g.add_argument("-q", "--quiet", action="store_true")

    t = p.add_argument_group("treino")
    t.add_argument("-e", "--episodes", type=int, default=None)
    t.add_argument("--max-steps", type=int, default=None,
                   help="default: derivado do pior caminho otimo do maze")
    t.add_argument("--lr",    type=float, default=None)
    t.add_argument("--gamma", type=float, default=None)
    t.add_argument("--batch", type=int,   default=None)
    t.add_argument("--learn", type=int,   default=None,
                   help="learn_interval base; multiplicado por (1+her_k) com HER")
    t.add_argument("--eps-decay-frac", type=float, default=None, metavar="F",
                   help="fracao do treino ate epsilon chegar em 0.1. "
                        "DIFERE POR MODELO no default (dense 0.5, reserved_sae "
                        "1.0) — iguale ao comparar modelos")
    t.add_argument("--min-replay", type=int, default=None, metavar="N",
                   help="transicoes minimas antes do 1o gradient step. "
                        "DIFERE POR MODELO no default — iguale ao comparar")

    s = p.add_argument_group("representacao de estado")
    s.add_argument("--encoder", choices=list(_ENCODERS.keys()), default=None)
    s.add_argument("--possible-actions", dest="possible_actions",
                   action="store_true", default=None,
                   help="liga o sensor de paredes vizinhas")
    s.add_argument("--no-possible-actions", dest="possible_actions",
                   action="store_false",
                   help="desliga o sensor de paredes vizinhas")
    s.add_argument("--last-states",  type=int, default=0)
    s.add_argument("--last-actions", type=int, default=0)

    d = p.add_argument_group("modelo denso (ignorado com --model reserved_sae)")
    d.add_argument("--hidden", type=int, nargs="+", default=None,
                   metavar="N", help="camadas ocultas, ex: --hidden 256 256")

    r = p.add_argument_group("ReservedSAECollab (ignorado com --model dense)")
    r.add_argument("--max-layers", type=int, default=None)
    r.add_argument("--sae-hidden", type=int, default=None, metavar="N",
                   help="largura ABSOLUTA da 1a camada oculta. Sem esta flag as "
                        "larguras sao proporcionais ao maze "
                        "(1/2 x acoes x linhas x colunas). Com ela, o modo "
                        "estatico e ativado e a largura nao depende do maze")
    r.add_argument("--sae-extra", type=int, default=None, metavar="N",
                   help="largura absoluta do extra branch (default: = --sae-hidden)")
    r.add_argument("--sae-width-delta", type=float, default=None, metavar="F",
                   help="fracao somada por camada em CRT/DRT/ALT "
                        "(default: 1/max_layers)")
    r.add_argument("--insertion",  choices=[x.value for x in LayerInsertionType],
                   default=None)
    r.add_argument("--layer-mode", choices=[x.name for x in LayerModeType],
                   default=None)
    r.add_argument("--mutation",   choices=[x.name for x in MutationMode],
                   default=None)
    r.add_argument("--new-layer-lr", type=float, default=None)
    r.add_argument("--insert-criterion",
                   choices=["plateau", "slope", "variance"],
                   default="plateau",
                   help="gatilho de insercao de camada. 'plateau': taxa no "
                        "held-out parou de subir (max vs max corrente). "
                        "'slope': tendencia entre medias de janelas disjuntas, "
                        "com warmup, cauda reservada e guarda de regressao. "
                        "'variance': criterio original do fast_experiment_1 - "
                        "praticamente nao dispara com start/goal dinamicos "
                        "(default: %(default)s)")
    r.add_argument("--insert-patience-evals", type=int, default=2,
                   help="[plateau/slope] tamanho da janela de avaliacoes "
                        "(default: %(default)s; use 4 com 'slope')")
    r.add_argument("--insert-min-delta", type=float, default=1.0,
                   help="[plateau/slope] pontos percentuais de ganho no held-out "
                        "que contam como progresso (default: %(default)s)")
    r.add_argument("--insert-warmup-frac", type=float, default=0.25, metavar="F",
                   help="[slope] fracao inicial do treino em que nenhuma camada "
                        "e inserida - a topologia minima precisa de tempo antes "
                        "de ser julgada (default: %(default)s)")
    r.add_argument("--insert-tail-frac", type=float, default=0.20, metavar="F",
                   help="[slope] fracao final do treino reservada para "
                        "consolidar a ultima ramificacao: nenhuma insercao "
                        "ocorre dentro dela (default: %(default)s)")
    r.add_argument("--insert-regress-guard", type=float, default=5.0, metavar="PP",
                   help="[slope] nao insere enquanto o held-out estiver mais de "
                        "PP pontos abaixo do melhor ja visto (default: "
                        "%(default)s)")
    r.add_argument("--no-insert-pace", dest="insert_pace", action="store_false",
                   help="[slope] desliga o escape por prazo. Sem ele um modelo "
                        "sem capacidade melhora devagar mas sem parar, nunca "
                        "aciona o gatilho de estagnacao e termina o treino sem "
                        "gastar o orcamento de camadas")
    p.set_defaults(insert_pace=True)
    r.add_argument("--insert-patience",     type=int,   default=None,
                   help="[variance] episodios entre checagens")
    r.add_argument("--insert-min-goals",    type=int,   default=None,
                   help="[variance] sucessos minimos na janela")
    r.add_argument("--insert-min-variance", type=float, default=None,
                   help="[variance] limiar do var_ratio")
    r.add_argument("--insert-skip-success", type=float, default=95.0,
                   help="acima desta taxa no held-out nenhuma camada e ativada "
                        "(default: %(default)s)")
    r.add_argument("--rolling-window",      type=int,   default=None)

    a = p.add_argument_group("algoritmo")
    a.add_argument("--no-her", dest="use_her", action="store_false",
                   help="desliga o Hindsight Experience Replay")
    a.add_argument("--her-k",  type=int, default=4,
                   help="goals ficticios por transicao (default: %(default)s)")
    a.add_argument("--n-step", type=int, default=3,
                   help="horizonte do retorno n-step; 1 = TD(0) "
                        "(default: %(default)s)")
    a.add_argument("--no-shaping", dest="use_shaping", action="store_false",
                   help="desliga o potential-based reward shaping")
    a.add_argument("--curriculum", action="store_true",
                   help="rampa o max_dist dos pares amostrados")

    v = p.add_argument_group("avaliacao")
    v.add_argument("--eval-pairs",    type=int, default=256,
                   help="tamanho do held-out (default: %(default)s)")
    v.add_argument("--eval-interval", type=int, default=50,
                   help="episodios entre avaliacoes (default: %(default)s)")

    return p


def print_defaults() -> None:
    for name, d in MODEL_DEFAULTS.items():
        print(f"\n[{name}]")
        for k, v in d.items():
            print(f"  {k:22s} {v}")
    print()


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.show_defaults:
        print_defaults()
        sys.exit(0)

    if args.selftest:
        sys.exit(0 if selftest(args.maze) else 1)

    # ── flags que nao fazem nada no modelo escolhido ────────────────────────
    # Um no-op silencioso e pior que um erro: o treino roda por horas com uma
    # configuracao diferente da que se pediu. Cada flag so vale para um dos
    # modelos, entao avisamos alto quando a combinacao nao faz sentido.
    DENSE_ONLY = ["hidden"]
    SAE_ONLY   = ["max_layers", "sae_hidden", "sae_extra", "sae_width_delta",
                  "insertion", "layer_mode", "mutation", "new_layer_lr",
                  "insert_patience", "insert_min_goals", "insert_min_variance"]

    ignored = [f for f in (SAE_ONLY if args.model == DENSE else DENSE_ONLY)
               if getattr(args, f, None) is not None]
    if ignored:
        flags = ", ".join("--" + f.replace("_", "-") for f in ignored)
        print(f"[AVISO] estas flags nao tem efeito com --model {args.model} "
              f"e serao IGNORADAS:")
        print(f"        {flags}")
        if args.model == RSAE and "hidden" in ignored:
            print("        As larguras do ReservedSAECollab vem de "
                  "gen_concrete_arch (multiplicadores da ModelArch x base_width).")
            print("        Para fixar a largura use --sae-hidden N "
                  "(ativa o modo estatico, como em fast_experiment_start_equal).")
        print()

    # Defaults dependentes do modelo: so preenchem o que ficou None.
    resolve_defaults(args.model, args)

    maze_tag = os.path.basename(args.maze).split(".")[0]
    out_dir  = args.out_dir or os.path.join(
        "dynamic_results", args.model, maze_tag, str(args.seed)
    )

    train_dynamic(
        maze_path           = args.maze,
        out_dir             = out_dir,
        model               = args.model,
        seed                = args.seed,
        episodes            = args.episodes,
        max_steps           = args.max_steps,
        encoder             = _ENCODERS[args.encoder],
        possible_actions    = args.possible_actions,
        num_last_states     = args.last_states,
        num_last_actions    = args.last_actions,
        hidden              = args.hidden or [256, 256],
        max_layers          = args.max_layers          or 4,
        sae_hidden          = args.sae_hidden,
        sae_extra           = args.sae_extra,
        sae_width_delta     = args.sae_width_delta,
        insertion           = args.insertion           or "CRT",
        layer_mode          = args.layer_mode          or "M4",
        mutation            = args.mutation            or "Hidden",
        new_layer_lr        = args.new_layer_lr        or 5e-5,
        insert_criterion    = args.insert_criterion,
        insert_patience     = args.insert_patience     or 15,
        insert_min_goals    = args.insert_min_goals    or 5,
        insert_min_variance = args.insert_min_variance or 0.6,
        insert_patience_evals = args.insert_patience_evals,
        insert_min_delta      = args.insert_min_delta,
        insert_skip_success = args.insert_skip_success,
        insert_warmup_frac    = args.insert_warmup_frac,
        insert_tail_frac      = args.insert_tail_frac,
        insert_regress_guard  = args.insert_regress_guard,
        insert_pace           = args.insert_pace,
        rolling_window      = args.rolling_window      or 20,
        lr                  = args.lr,
        gamma               = args.gamma,
        batch_size          = args.batch,
        learn_interval      = args.learn,
        use_her             = args.use_her,
        her_k               = args.her_k,
        n_step              = args.n_step,
        use_shaping         = args.use_shaping,
        eps_decay_frac      = args.eps_decay_frac,
        min_replay          = args.min_replay,
        eval_pairs          = args.eval_pairs,
        eval_interval       = args.eval_interval,
        curriculum          = args.curriculum,
        device              = args.device,
        verbose             = not args.quiet,
    )
