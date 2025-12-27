#python3 torch_train.py

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import os

from env.MazeEnv import *
from env.MazeWrapper import StateEncoder,MazeGymWrapper

from models.QModels import TorchDDQN, exp_decay_factor_to
from models.AStar import AStarQModel,a_star_maze_solve
from models.Path import Path,get_best_path
from models.QTable import genearate_qtable_from_model

file_path = sys.argv[-1] if len(sys.argv) > 1 else ""
if not os.path.exists(file_path) or file_path == '':
    print(f"[ERROR] File Not Found: {file_path}")
    print("Usage: python3 torch_train.py <maze_file.maze>")
    exit(-1)
base_name = os.path.basename(file_path).split(".")[0]

raw_env = MazeEnv(file_path,rewards_scaled=False,
                  pass_through_walls=False)
print(f"\n{'='*70}")
print(f"MAZE: {base_name}")
print(f"Size: {raw_env.rows}x{raw_env.cols} ({raw_env.rows * raw_env.cols} células)")
print(f"Start: {raw_env.agent_start}")
print(f"Goal: {raw_env.agent_goal}")
print(f"{'='*70}\n")

train_state_encoder = StateEncoder.COORDS
env = MazeGymWrapper(raw_env, train_state_encoder,
                     num_last_states=1,  
                     possible_actions_feature=True,
                     visited_count=False
                     )

EPISODES    = 200
STATE_SIZE  = env.state_size
ACTION_SIZE = env.action_size
MAX_STEPS   = env.maze.opens_count * ACTION_SIZE
HIDDEN_SIZE = env.rows * env.cols * env.action_size
GAMMA = 0.99
LR = 1e-5

BATCH_SIZE  = 1024
LEARN_INTERVAL = 4
LOG_INTERVAL= 10

BUFFER_SIZE = 100_000
TOTAL_STEPS = MAX_STEPS * EPISODES

epsilon_decay = exp_decay_factor_to(
    final_epsilon=0.1,
    final_step=TOTAL_STEPS,
    epsilon_start=1.0,
    convergence_threshold=0.01
)

print("[INFO] Resolvendo com A*...")
a_star_path = a_star_maze_solve(raw_env)
a_star_model = AStarQModel(env)
print(f"[INFO] A* path length: {a_star_path.len}\n")

policy_arq = nn.Sequential(
    nn.Linear(STATE_SIZE, HIDDEN_SIZE),
    nn.ReLU(),               
    nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2),
    nn.ReLU(),
    nn.Linear(HIDDEN_SIZE // 2, ACTION_SIZE),
)

print(f"{'='*70}\n")
print(f"[INFO] Arquitetura da rede:")
total_params = sum(p.numel() for p in policy_arq.parameters())
print(f"  Layers: {STATE_SIZE} → {HIDDEN_SIZE} → {HIDDEN_SIZE // 2} → {ACTION_SIZE}")
print(f"  Total params: {total_params:,}")
print(f"{'='*70}\n")

agent = TorchDDQN(
    sequential_list=policy_arq,
    state_size=STATE_SIZE,
    action_size=ACTION_SIZE,
    lr=LR,
    gamma=GAMMA, 
    batch_size=BATCH_SIZE,
    epsilon_start=1.0,
    learn_interval=LEARN_INTERVAL,
    epsilon_final=0.1,
    tau=0.005,
    grad_clip= 10.0,
    min_replay_size=max(1000,2*BATCH_SIZE),
    epsilon_decay=epsilon_decay,
    target_update=int(TOTAL_STEPS / 50),
    buffer_size=BUFFER_SIZE
)

print(f"{'='*70}\n")
print(f"[INFO] Hiperparâmetros do agente:")
print(f"  Learning rate: {LR}")
print(f"  Gamma: {GAMMA}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Buffer size: {BUFFER_SIZE}")
print(f"  Target update: {agent.target_update} steps")
print(f"  Epsilon: {agent.epsilon_start} → {agent.epsilon_final}")
print(f"{'='*70}\n")

print(f"[INFO] Iniciando treinamento...\n")

rewards_per_episode = []
goals_count = 0

for ep in range(1,EPISODES+1):
    total_reward = 0.0
    state = env.reset()
    # MAKING STATE BE (1,NUM_FEATURES)
    state = state.reshape(1, -1)
    
    for step in range(MAX_STEPS):
        action = agent.act(state, eval=False)
        next_state, reward, done, extras_dict = env.step(action)
        if env.isGoal(extras_dict["raw_ns"]):
            goals_count += 1
        # MAKING NEW STATE BE (1,NUM_FEATURES)
        next_state.reshape(1,-1)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done: break
    
    agent.update_epsilon()
    rewards_per_episode.append(total_reward)
    if ep % LOG_INTERVAL == 0:
        recent = np.mean(rewards_per_episode[-LOG_INTERVAL:])
        print(f"Ep {ep}/{EPISODES} goal_cnt={goals_count} loss={agent.loss:e} reward={total_reward:.3f}  recent_avg({LOG_INTERVAL})={recent:.3f}  eps={agent.epsilon:.3f}")

print(f"\n[INFO] Treinamento concluído!")
print(f"  Total steps executados: {agent.steps_done}")
print(f"  Epsilon final: {agent.epsilon:.4f}")

qtable_path = f"qtables/{base_name}_torch.qtable"
genearate_qtable_from_model(
    env, agent, ACTION_SIZE,
    get_q_val_method='__call__'
).save(qtable_path)
print(f"\n[INFO] Q-table salva em: {qtable_path}")

print(f"\n[INFO] Gerando melhor caminho...")
model_path:Path = get_best_path(env, agent, max_steps=MAX_STEPS)
print(model_path)

qtable_path = f"qtables/{base_name}_astar.qtable"
genearate_qtable_from_model(
    env, a_star_model, ACTION_SIZE,
    get_q_val_method='__call__',
    use_extras=False
).save(qtable_path)
print(f"\n[INFO] A* Q-table salva em: {qtable_path}")

print("Model Path len: ",model_path.len)
print("A* Path len: ",a_star_path.len)

if model_path.len > 0 and a_star_path.len > 0:
    similarity = model_path.similarity_to(a_star_path)
    print(f"Similaridade: {similarity:.1%}")