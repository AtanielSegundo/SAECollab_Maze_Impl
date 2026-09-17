from typing import *

import os
import sys
import json
import argparse

import numpy as np
import torch
import pyray as pr

from dynamic_maze_start_goal import (
    MazeTopology, DynamicGoalMazeEnv, TorchDDQN, load_trained_run, q_values,
)


"""
Viewer interativo para os agentes goal-conditioned treinados por
dynamic_maze_start_goal.py.

Clique para posicionar start e goal em qualquer celula navegavel, aperte PLAY,
e o agente treinado resolve. Inspirado no c_qlearning/src/agentViewer.c, mas
sem editor de maze, sem camera e sem dialogo de arquivo.

O limite de passos e rows * cols (o tamanho do labirinto), como pedido.

CONTROLES
    botao esquerdo    posiciona o START
    botao direito     posiciona o GOAL
    SPACE             play / pause
    N                 um passo (funciona pausado)
    R                 volta o agente ao start
    C                 caminho otimo do BFS por cima (ground truth)
    T                 rastro do agente
    P                 setas da politica gulosa em cada celula
    LEFT / RIGHT      mais lento / mais rapido
    ESC               sair

USO
    python dynamic_maze_viewer.py -d dynamic_results/small_eg/333
    python dynamic_maze_viewer.py -d <dir> --model model_last.pth
    python dynamic_maze_viewer.py -h
"""


WINDOW_W = 1280
WINDOW_H = 720
HUD_H    = 118
PAD      = 16

C_BG       = pr.Color(24,  26,  32,  255)
C_WALL     = pr.Color(52,  56,  68,  255)
C_OPEN     = pr.Color(232, 234, 240, 255)
C_GRID     = pr.Color(200, 203, 212, 255)
C_START    = pr.Color(64,  190, 120, 255)
C_GOAL     = pr.Color(226, 78,  78,  255)
C_AGENT    = pr.Color(250, 204, 62,  255)
C_TRAIL    = pr.Color(250, 204, 62,  90)
C_OPT      = pr.Color(90,  150, 240, 110)
C_ARROW    = pr.Color(120, 124, 140, 255)
C_TXT      = pr.Color(238, 240, 246, 255)
C_DIM      = pr.Color(150, 154, 166, 255)
C_OK       = pr.Color(96,  214, 140, 255)
C_FAIL     = pr.Color(238, 104, 104, 255)

SPEEDS = [0.02, 0.05, 0.10, 0.20, 0.40, 0.80]   # segundos por passo


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento
# ─────────────────────────────────────────────────────────────────────────────

def load_run(run_dir: str, model_name: str, maze_override: Optional[str],
             device: str):
    """Wrapper de load_trained_run() com mensagens de erro amigaveis.

    A reconstrucao em si vive em dynamic_maze_start_goal para que viewer e
    dynamic_compare carreguem checkpoints exatamente do mesmo jeito.
    """
    try:
        return load_trained_run(run_dir, model_name, maze_override,
                                device, verbose=True)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        if "summary.json" in str(e):
            print("        O viewer precisa dele para reconstruir a mesma rede.")
            print("        Rode um treino novo ou aponte -d para o diretorio certo.")
        sys.exit(-1)
    except Exception as e:
        print("[ERROR] nao consegui carregar o treino:")
        print(f"  {e}")
        sys.exit(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Layout do grid
# ─────────────────────────────────────────────────────────────────────────────

class GridView:
    """Converte celula <-> pixel, com a celula dimensionada para caber na
    janela (o maze inteiro sempre visivel, sem camera nem scroll)."""

    def __init__(self, rows: int, cols: int):
        self.rows, self.cols = rows, cols
        self.recompute()

    def recompute(self) -> None:
        w = pr.get_screen_width()  - 2 * PAD
        h = pr.get_screen_height() - HUD_H - 2 * PAD
        self.cell = max(4, int(min(w / self.cols, h / self.rows)))
        gw = self.cell * self.cols
        gh = self.cell * self.rows
        self.ox = PAD + (w - gw) // 2
        self.oy = PAD + (h - gh) // 2

    def cell_rect(self, r: int, c: int) -> Tuple[int, int, int, int]:
        return self.ox + c * self.cell, self.oy + r * self.cell, self.cell, self.cell

    def center(self, r: int, c: int) -> Tuple[int, int]:
        return (self.ox + c * self.cell + self.cell // 2,
                self.oy + r * self.cell + self.cell // 2)

    def mouse_cell(self) -> Optional[Tuple[int, int]]:
        m = pr.get_mouse_position()
        c = int((m.x - self.ox) // self.cell)
        r = int((m.y - self.oy) // self.cell)
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return r, c
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Estado do episodio
# ─────────────────────────────────────────────────────────────────────────────

class Episode:
    """Encapsula o rollout: quem chama so faz reset() e step()."""

    def __init__(self, env: DynamicGoalMazeEnv, topo: MazeTopology,
                 agent: TorchDDQN, max_steps: int):
        self.env, self.topo, self.agent = env, topo, agent
        self.max_steps = max_steps
        self.start_idx = int(topo.nav[0])
        self.goal_idx  = int(topo.nav[-1])
        self.reset()

    def reset(self) -> None:
        self.state   = self.env.reset(self.start_idx, self.goal_idx)
        self.steps   = 0
        self.trail   = [self.start_idx]
        self.done    = False        # chegou no goal
        self.dead    = False        # estourou o limite ou entrou em ciclo
        self.reason  = ""
        self.visited = {self.start_idx}
        self.reward  = 0.0

    def set_start(self, idx: int) -> None:
        if idx == self.goal_idx:
            return
        self.start_idx = idx
        self.reset()

    def set_goal(self, idx: int) -> None:
        if idx == self.start_idx:
            return
        self.goal_idx = idx
        self.reset()

    @property
    def finished(self) -> bool:
        return self.done or self.dead

    @property
    def optimal(self) -> int:
        return self.topo.optimal(self.start_idx, self.goal_idx)

    def step(self) -> None:
        if self.finished:
            return
        with torch.no_grad():
            action = self.agent.act(self.state, eval=True)
        self.state, r, done, info = self.env.step(action)
        self.steps  += 1
        self.reward += r
        nxt = int(info["next_idx"])
        self.trail.append(nxt)

        if done:
            self.done   = True
            self.reason = "GOAL"
            return
        # Politica gulosa em ambiente deterministico: repetir uma celula
        # significa ciclo infinito. Detectar aqui evita esperar o limite.
        if nxt in self.visited:
            self.dead   = True
            self.reason = "LOOP"
            return
        self.visited.add(nxt)
        if self.steps >= self.max_steps:
            self.dead   = True
            self.reason = "LIMITE"


# ─────────────────────────────────────────────────────────────────────────────
# Desenho
# ─────────────────────────────────────────────────────────────────────────────

def draw_maze(gv: GridView, topo: MazeTopology) -> None:
    for r in range(topo.rows):
        for c in range(topo.cols):
            x, y, w, h = gv.cell_rect(r, c)
            wall = not topo.walkable[r * topo.cols + c]
            pr.draw_rectangle(x, y, w, h, C_WALL if wall else C_OPEN)
            if not wall and gv.cell >= 10:
                pr.draw_rectangle_lines(x, y, w, h, C_GRID)


def draw_optimal(gv: GridView, topo: MazeTopology, start: int, goal: int) -> None:
    """Reconstroi um caminho minimo descendo o gradiente de dist[.., goal]."""
    d = topo.dist[:, goal]
    cur, guard = start, 0
    while cur != goal and guard < topo.n:
        guard += 1
        nxts = [int(v) for v in topo.nbr[cur] if v >= 0 and d[v] == d[cur] - 1]
        if not nxts:
            break
        cur = nxts[0]
        r, c = topo.rc(cur)
        x, y, w, h = gv.cell_rect(r, c)
        pr.draw_rectangle(x, y, w, h, C_OPT)


def draw_trail(gv: GridView, topo: MazeTopology, trail: Sequence[int]) -> None:
    for idx in trail:
        r, c = topo.rc(idx)
        x, y, w, h = gv.cell_rect(r, c)
        pad = max(1, gv.cell // 4)
        pr.draw_rectangle(x + pad, y + pad, w - 2 * pad, h - 2 * pad, C_TRAIL)


def draw_policy(gv: GridView, topo: MazeTopology, env: DynamicGoalMazeEnv,
                agent: TorchDDQN, goal: int) -> None:
    """Seta da acao gulosa em cada celula navegavel, para o goal atual.

    Monta as observacoes de todas as celulas de uma vez (build_obs ignora o
    historico, o que e exatamente o que se quer para um mapa estatico) e faz
    um unico forward.
    """
    nav = topo.nav
    obs = np.stack([env.build_obs(int(i), goal) for i in nav])
    with torch.no_grad():
        t = torch.from_numpy(obs).to(agent.device)
        acts = q_values(agent.policy_net, t).argmax(dim=1).cpu().numpy()

    from dynamic_maze_start_goal import _ADR, _ADC
    ln = max(2, gv.cell // 3)
    for i, a in zip(nav, acts):
        idx = int(i)
        if idx == goal:
            continue
        r, c = topo.rc(idx)
        cx, cy = gv.center(r, c)
        dr, dc = int(_ADR[a]), int(_ADC[a])
        pr.draw_line(cx, cy, cx + dc * ln, cy + dr * ln, C_ARROW)


def draw_hud(ep: Episode, playing: bool, speed_i: int, info: Dict[str, Any],
             flags: Dict[str, bool]) -> None:
    y0 = pr.get_screen_height() - HUD_H
    pr.draw_rectangle(0, y0, pr.get_screen_width(), HUD_H, pr.Color(18, 20, 25, 255))
    pr.draw_line(0, y0, pr.get_screen_width(), y0, pr.Color(60, 64, 76, 255))

    opt = ep.optimal
    if ep.done:
        status, col = f"GOAL em {ep.steps} passos", C_OK
        if opt > 0:
            status += f"   (otimo {opt}, razao {ep.steps / opt:.2f})"
    elif ep.dead:
        status, col = f"FALHOU [{ep.reason}] apos {ep.steps} passos", C_FAIL
    elif playing:
        status, col = "rodando...", C_TXT
    else:
        status, col = "pausado  —  SPACE para dar play", C_DIM

    pr.draw_text(status, PAD, y0 + 10, 22, col)

    line2 = (f"passos {ep.steps}/{ep.max_steps}    otimo {opt if opt > 0 else '-'}"
             f"    reward {ep.reward:+.2f}    vel {SPEEDS[speed_i]*1000:.0f}ms")
    pr.draw_text(line2, PAD, y0 + 40, 18, C_DIM)

    extra = ""
    if info.get("model") == "reserved_sae":
        extra = (f"  |  {info.get('active_layers','?')}/{info.get('max_layers','?')} camadas"
                 f" {info.get('insertion','')}/{info.get('layer_mode','')}"
                 f"/{info.get('mutation','')}")
    line3 = (f"{info.get('model', 'dense')}  |  {info['maze']}  |  {info['encoder']}"
             f"  |  n-step {info.get('n_step','?')}"
             f"  |  HER {'on' if info.get('use_her') else 'off'}{extra}"
             f"  |  {os.path.basename(info['model_path'])}")
    pr.draw_text(line3, PAD, y0 + 62, 16, C_DIM)

    on = lambda b: "on" if b else "off"
    line4 = (f"L-click start   R-click goal   SPACE play   N passo   R reset   "
             f"C otimo[{on(flags['opt'])}]   T rastro[{on(flags['trail'])}]   "
             f"P politica[{on(flags['pol'])}]   <- -> vel")
    pr.draw_text(line4, PAD, y0 + 84, 16, pr.Color(120, 124, 140, 255))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dynamic_maze_viewer.py",
        description="Viewer interativo para agentes goal-conditioned treinados "
                    "por dynamic_maze_start_goal.py.",
        epilog="""
CONTROLES
  botao esquerdo   posiciona o START
  botao direito    posiciona o GOAL
  SPACE            play / pause
  N                um passo
  R                reset
  C                caminho otimo (BFS)
  T                rastro
  P                setas da politica
  LEFT / RIGHT     mais lento / mais rapido
  ESC              sair
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-d", "--dir", dest="run_dir", required=True,
                   help="diretorio do treino (com summary.json e o .pth)")
    p.add_argument("--model", dest="model_name", default="model_last.pth",
                   help="arquivo do modelo (default: %(default)s)")
    p.add_argument("-m", "--maze", default=None,
                   help="sobrescreve o maze do summary.json")
    p.add_argument("--device", choices=["cpu", "cuda"], default=None)
    return p


if __name__ == "__main__":

    args = build_parser().parse_args()

    run_dir    = args.run_dir
    model_name = args.model_name
    maze_over  = args.maze
    device     = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(run_dir):
        print(f"[ERROR] diretorio nao existe: {run_dir}")
        sys.exit(-1)

    maze, topo, env, agent, summary, model_path = load_run(
        run_dir, model_name, maze_over, device
    )

    # Limite de passos = tamanho do labirinto, como pedido.
    max_steps = topo.rows * topo.cols

    print(f"[VIEWER] {maze.tag} {topo.rows}x{topo.cols}  "
          f"limite de passos = {max_steps}")
    print(f"[VIEWER] modelo: {model_path}")

    pr.set_config_flags(pr.ConfigFlags.FLAG_WINDOW_RESIZABLE)
    pr.set_trace_log_level(pr.LOG_ERROR);
    pr.init_window(WINDOW_W, WINDOW_H, f"Dynamic Maze Viewer — {maze.tag}")
    pr.set_target_fps(60)

    gv = GridView(topo.rows, topo.cols)
    ep = Episode(env, topo, agent, max_steps)

    playing  = False
    speed_i  = 2
    acc      = 0.0
    flags    = {"opt": True, "trail": True, "pol": False}
    info     = dict(summary)
    info["model_path"] = model_path

    while not pr.window_should_close():
        if pr.is_window_resized():
            gv.recompute()

        # ── input ───────────────────────────────────────────────────────────
        if pr.is_mouse_button_pressed(pr.MOUSE_BUTTON_LEFT):
            cell = gv.mouse_cell()
            if cell and topo.walkable[cell[0] * topo.cols + cell[1]]:
                ep.set_start(topo.idx(*cell))
                playing = False
        if pr.is_mouse_button_pressed(pr.MOUSE_BUTTON_RIGHT):
            cell = gv.mouse_cell()
            if cell and topo.walkable[cell[0] * topo.cols + cell[1]]:
                ep.set_goal(topo.idx(*cell))
                playing = False

        if pr.is_key_pressed(pr.KEY_SPACE):
            if ep.finished:
                ep.reset()
            playing = not playing
        if pr.is_key_pressed(pr.KEY_N):
            playing = False
            ep.step()
        if pr.is_key_pressed(pr.KEY_R):
            ep.reset()
            playing = False
        if pr.is_key_pressed(pr.KEY_C):
            flags["opt"] = not flags["opt"]
        if pr.is_key_pressed(pr.KEY_T):
            flags["trail"] = not flags["trail"]
        if pr.is_key_pressed(pr.KEY_P):
            flags["pol"] = not flags["pol"]
        if pr.is_key_pressed(pr.KEY_RIGHT):
            speed_i = max(0, speed_i - 1)
        if pr.is_key_pressed(pr.KEY_LEFT):
            speed_i = min(len(SPEEDS) - 1, speed_i + 1)

        # ── update ──────────────────────────────────────────────────────────
        if playing and not ep.finished:
            acc += pr.get_frame_time()
            while acc >= SPEEDS[speed_i] and not ep.finished:
                acc -= SPEEDS[speed_i]
                ep.step()
        else:
            acc = 0.0
        if ep.finished:
            playing = False

        # ── draw ────────────────────────────────────────────────────────────
        pr.begin_drawing()
        pr.clear_background(C_BG)

        draw_maze(gv, topo)
        if flags["opt"]:
            draw_optimal(gv, topo, ep.start_idx, ep.goal_idx)
        if flags["pol"]:
            draw_policy(gv, topo, env, agent, ep.goal_idx)
        if flags["trail"]:
            draw_trail(gv, topo, ep.trail)

        sr, sc = topo.rc(ep.start_idx)
        x, y, w, h = gv.cell_rect(sr, sc)
        pr.draw_rectangle(x, y, w, h, C_START)
        gr, gc = topo.rc(ep.goal_idx)
        x, y, w, h = gv.cell_rect(gr, gc)
        pr.draw_rectangle(x, y, w, h, C_GOAL)

        ar, ac = topo.rc(ep.trail[-1])
        cx, cy = gv.center(ar, ac)
        pr.draw_circle(cx, cy, max(2.0, gv.cell * 0.34), C_AGENT)

        draw_hud(ep, playing, speed_i, info, flags)
        pr.end_drawing()

    pr.close_window()

