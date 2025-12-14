import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import sys
import os

from env.MazeEnv import *
from env.MazeWrapper import StateEncoder, MazeGymWrapper

from models.QModels import SAECollabDDQN, exp_decay_factor_to
from models.AStar import AStarQModel, a_star_maze_solve
from models.Path import get_best_path
from models.QTable import genearate_qtable_from_model
from StackedCollab.collabNet import MutationMode

# ============================================================================
# CARREGAMENTO DO MAZE
# ============================================================================
file_path = sys.argv[-1] if len(sys.argv) > 1 else ""
if not os.path.exists(file_path) or file_path == '':
    print(f"[ERROR] File Not Found: {file_path}")
    print("Usage: python3 sae_train.py <maze_file.maze>")
    exit(-1)
base_name = os.path.basename(file_path).split(".")[0]

raw_env = MazeEnv(file_path, rewards_scaled=False, pass_through_walls=True)
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
                     visited_count=True
                     )

# ============================================================================
# HIPERPARÂMETROS
# ============================================================================
EPISODES    = 300
STATE_SIZE  = env.state_size
ACTION_SIZE = env.action_size
MAX_STEPS   = env.maze.opens_count * env.action_size

_HIDDEN_SIZE  = env.rows * env.cols * env.action_size

INITIAL_HIDDEN    = max(128,_HIDDEN_SIZE)
NEW_BRANCH_HIDDEN = max(16,_HIDDEN_SIZE // 4)
EXTRA_HIDDEN      = max(16, _HIDDEN_SIZE // 4)
LAYER_MUTATION_MODE = MutationMode.Hidden

HIDDEN_FN = nn.GELU()
EXTRA_FN  = nn.GELU()
TARGET_FN = HIDDEN_FN

NUM_BRANCHES      = 4
ADD_BRANCH_MOD    = EPISODES // (NUM_BRANCHES + 1)

GAMMA = 0.99
LR = [1e-5,5e-5]
BATCH_SIZE = 256
LEARN_INTERVAL = 4
LOG_INTERVAL = 10
BUFFER_SIZE = 100_000
TOTAL_STEPS = MAX_STEPS * EPISODES

epsilon_decay = exp_decay_factor_to(
    final_epsilon=0.1,
    final_step=TOTAL_STEPS,
    epsilon_start=1.0,
    convergence_threshold=0.01
)

TEST_INTERVAL = 5

# ============================================================================
# REFERÊNCIA: A*
# ============================================================================
print("[INFO] Resolvendo com A*...")
a_star_path = a_star_maze_solve(raw_env)
a_star_model = AStarQModel(env)
print(f"[INFO] A* path length: {a_star_path.len}\n")

# ============================================================================
# SAECollabDDQN
# ============================================================================
print(f"{'='*70}")
print("[INFO] Criando SAECollabDDQN...")

sae_agent = SAECollabDDQN(
    state_size=STATE_SIZE,
    action_size=ACTION_SIZE,
    first_hidden_size=INITIAL_HIDDEN,
    hidden_activation=HIDDEN_FN,
    out_activation=nn.Identity(),
    accelerate_etas=True,
    lr=LR,
    gamma=GAMMA,
    batch_size=BATCH_SIZE,
    buffer_size=BUFFER_SIZE,
    epsilon_start=1.0,
    epsilon_final=0.1,
    epsilon_decay=epsilon_decay,
    target_update=int(TOTAL_STEPS / 50),
    learn_interval=LEARN_INTERVAL,
    tau=0.005,
    grad_clip=10.0,
    min_replay_size=1000
)

sae_initial_params = sum(p.numel() for p in sae_agent.policy_net.parameters())
print(f"  Input size: {STATE_SIZE}")
print(f"  Outpu size: {ACTION_SIZE}")
print(f"  Initial hidden: {INITIAL_HIDDEN}")
print(f"  Initial params: {sae_initial_params:,}")
print(f"  Num branches: {NUM_BRANCHES}")
print(f"{'='*70}\n")

sae_history = {
    "rewards": [],
    "goals": [],
    "recent_avg": [],
    "variance_ratio": [],
    "branch_changes": [],
    "loss": []
}

branches_added = 0
current_layer = 0

# Flag global que impede adicionar novas branches quando for encontrado caminho
WAY_TO_GOAL_FOUND = False

# ============================================================================
# Função de avaliação (teste)
# - define agent em modo eval, força epsilon=0.0 temporariamente
# - executa um episódio greedy (sem grad) e retorna True se achar caminho até o goal
# ============================================================================
def eval_agent_find_path(agent, env, max_steps):
    """
    Executa um episódio em modo determinístico (epsilon=0) para verificar se o agente
    consegue alcançar o objetivo. Não altera permanentemente o agente.
    Retorna True se goal for alcançado, False caso contrário.
    """
    # salva estados anteriores
    old_epsilon = getattr(agent, "epsilon", None)
    # define modo eval
    prev_training = getattr(agent.policy_net, "training", True)
    agent.policy_net.eval()
    # força epsilon 0
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0

    ns = env.reset()
    state = ns.reshape(1, -1)
    reached = False

    with torch.no_grad():
        for _ in range(max_steps):
            try:
                action = agent.act(state, eval=True)
            except TypeError:
                action = agent.act(state)  # fallback
            next_state, reward, done, extras = env.step(action)
            if env.isGoal(extras.get("raw_ns", extras)):
                reached = True
                break
            state = next_state.reshape(1, -1)
            if done:
                break

    if hasattr(agent, "epsilon") and old_epsilon is not None:
        agent.epsilon = old_epsilon
    if prev_training:
        agent.policy_net.train()
    return reached

print(f"[INFO] Iniciando treinamento...\n")
print(f"{'Episode':>7} {'Goals':>6} {'Reward':>8} {'Avg':>8} {'Var/Mean':>9} {'Loss':>10} {'Eps':>6} {'Layer':>5} ")
print(f"{'-'*80}")

for ep in range(1, EPISODES + 1):

    state = env.reset()
    state = state.reshape(1, -1)
    sae_total_reward = 0.0
    sae_goal_reached = False
    
    for step in range(MAX_STEPS):
        action = sae_agent.act(state, eval=False)
        next_state, reward, done, extras_dict = env.step(action)
        
        if env.isGoal(extras_dict["raw_ns"]):
            sae_goal_reached = True
        
        next_state = next_state.reshape(1, -1)
        sae_agent.remember(state, action, reward, next_state, done)
        state = next_state
        sae_total_reward += reward
        
        if done:
            break
    
    sae_agent.policy_net.step_all_etas()
    sae_agent.update_epsilon()

    # registra reward/goals
    sae_history["rewards"].append(sae_total_reward)
    sae_history["goals"].append(1 if sae_goal_reached else 0)

    # registra loss convertendo para float (garante compatibilidade CPU)
    loss_value = getattr(sae_agent, "loss", None)
    if torch.is_tensor(loss_value):
        loss_value = loss_value.detach().cpu().item()
    else:
        try:
            loss_value = float(loss_value)
        except Exception:
            loss_value = np.nan
    sae_history["loss"].append(loss_value)

    # LOG
    if ep % LOG_INTERVAL == 0:
        sae_goals_sum = sum(sae_history["goals"][-LOG_INTERVAL:])
        window_rewards = sae_history["rewards"][-LOG_INTERVAL:]
        window_mean = np.mean(window_rewards)
        window_var = np.var(window_rewards)
        var_ratio = window_var / abs(window_mean) if abs(window_mean) > 1e-6 else float('inf')

        print(f"{ep:7d} {sae_goals_sum:6d} {sae_total_reward:8.3f} "
                f"{window_mean:8.3f} {var_ratio:9.4f} "
                f"{loss_value:10.3e} {sae_agent.epsilon:6.3f} {current_layer:5d}"
                )

    # AVALIAR PERIODICAMENTE se agente já aprendeu caminho
    if (ep % TEST_INTERVAL == 0) and (not WAY_TO_GOAL_FOUND):
        found = eval_agent_find_path(sae_agent, env, MAX_STEPS)
        if found:
            WAY_TO_GOAL_FOUND = True
            sae_path = get_best_path(env, sae_agent, max_steps=MAX_STEPS)
            print(f"[INFO] WAY_TO_GOAL_FOUND ativado no episódio {ep} — agente encontrou um caminho determinístico até o goal.")
            print(sae_path)


    # ADICIONAR BRANCH APENAS SE NÃO HOUVER WAY_TO_GOAL_FOUND
    if (ep % ADD_BRANCH_MOD == 0) and (branches_added + 1 <= NUM_BRANCHES) and (not WAY_TO_GOAL_FOUND):
        window_rewards = sae_history["rewards"][-ADD_BRANCH_MOD:]
        window_mean = np.mean(window_rewards)
        window_var = np.var(window_rewards)
        var_ratio = window_var / abs(window_mean) if abs(window_mean) > 1e-6 else float('inf')

        sae_history["variance_ratio"].append(var_ratio)

        print(f"\n{'='*70}")
        print(f"  Window mean: {window_mean:.3f}")
        print(f"  Window variance: {window_var:.3f}")
        print(f"  Var/Mean ratio: {var_ratio:.4f}")
        print(f"[INFO] Adicionando nova branch (total: {branches_added + 1}/{NUM_BRANCHES})...")
        print(f"\n{'='*70}")

        sae_agent.add_layer(
            layer_hidden_size=NEW_BRANCH_HIDDEN,
            layer_extra_size=EXTRA_HIDDEN,
            k=1.0,
            mutation_mode=LAYER_MUTATION_MODE,
            target_fn=TARGET_FN,
            eta=0.0,
            eta_increment=1 / ADD_BRANCH_MOD,
            hidden_activation=HIDDEN_FN,
            out_activation=nn.Identity(),
            extra_activation=EXTRA_FN,
        )
        
        current_layer = len(sae_agent.policy_net.layers) - 1
        branches_added += 1
        
        new_params = sum(p.numel() for p in sae_agent.policy_net.parameters())
        print(f"[INFO] Nova camada {current_layer} adicionada.")
        print(f"[INFO] Parâmetros: {new_params:,} (+{new_params - sae_initial_params:,})")
        print(f"{'='*70}\n")

        sae_history["branch_changes"].append({
            "episode": ep,
            "layer": current_layer,
            "params": new_params,
            "var_ratio": var_ratio,
            "mean_reward": window_mean
        })

# ============================================================================
# SALVAMENTO DOS MODELOS
# ============================================================================
print(f"\n[INFO] Treinamento concluído!")
print(f"  SAE Total steps: {sae_agent.steps_done}")
print(f"  SAE Branches added: {branches_added}")
print(f"  SAE Final params: {sum(p.numel() for p in sae_agent.policy_net.parameters()):,}")

os.makedirs("models", exist_ok=True)
os.makedirs("qtables", exist_ok=True)

sae_agent.save(f"models/{base_name}_sae_collab.pth")
print(f"\n[INFO] Modelos salvos em models/")

# ============================================================================
# GERAÇÃO DE Q-TABLES E PATHS
# ============================================================================
print(f"\n[INFO] Gerando Q-tables e paths...")

sae_path = get_best_path(env, sae_agent, max_steps=MAX_STEPS)
print(f"\nSAE Path: {sae_path}")

genearate_qtable_from_model(
    env, sae_agent, ACTION_SIZE,
    get_q_val_method='__call__'
).save(f"qtables/{base_name}_sae_collab.qtable")

genearate_qtable_from_model(
    env, a_star_model, ACTION_SIZE,
    get_q_val_method='__call__',
    use_extras=False
).save(f"qtables/{base_name}_astar.qtable")

print(f"\n[INFO] Q-tables salvas em qtables/")

# ============================================================================
# COMPARAÇÃO DE RESULTADOS
# ============================================================================
print(f"\n{'='*70}")
print("COMPARAÇÃO DE RESULTADOS")
print(f"{'='*70}")
print(f"{'Modelo':<20} {'Path Len':>10} {'Sim. A*':>10} {'Final Params':>15}")
print(f"{'-'*70}")

if sae_path.len > 0:
    sae_sim = sae_path.similarity_to(a_star_path)
    print(f"{'SAE ColabNet':<20} {sae_path.len:>10} {sae_sim:>9.1%} "
          f"{sum(p.numel() for p in sae_agent.policy_net.parameters()):>15,}")
else:
    print(f"{'SAE ColabNet':<20} {'FAILED':>10} {'N/A':>10} "
          f"{sum(p.numel() for p in sae_agent.policy_net.parameters()):>15,}")

print(f"{'A* (Optimal)':<20} {a_star_path.len:>10} {'100.0%':>10} {'N/A':>15}")
print(f"{'='*70}\n")

# ============================================================================
# VISUALIZAÇÃO
# ============================================================================
fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

episodes_range = range(1, EPISODES + 1)

# Rewards
ax1 = axes[0]
ax1.plot(episodes_range, sae_history["rewards"], label="Raw Rewards", alpha=0.3)

if len(sae_history["rewards"]) >= ADD_BRANCH_MOD:
    moving_avg = [np.mean(sae_history["rewards"][max(0, i-ADD_BRANCH_MOD):i+1]) 
                  for i in range(len(sae_history["rewards"]))]
    ax1.plot(episodes_range, moving_avg, label=f"Moving Avg ({ADD_BRANCH_MOD} ep)", linewidth=2)

ax1.set_ylabel("Reward")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_title("Training Progress: SAE ColabNet with Variance-based Branching")

# Variance / Mean
ax2 = axes[1]
if len(sae_history["variance_ratio"]) > 0:
    var_episodes = [ADD_BRANCH_MOD * (i + 1) for i in range(len(sae_history["variance_ratio"]))]
    ax2.plot(var_episodes, sae_history["variance_ratio"], label="Var/Mean Ratio", linewidth=1.5, marker='o')
    ax2.set_ylabel("Variance / Mean")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

# Goals acumulados
ax3 = axes[2]
sae_cumulative_goals = np.cumsum(sae_history["goals"])
ax3.plot(episodes_range, sae_cumulative_goals, label="Cumulative Goals")
ax3.set_ylabel("Cumulative Goals")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Success rate
ax4 = axes[3]
window = LOG_INTERVAL
if len(sae_history["goals"]) >= window:
    success_rate = [np.mean(sae_history["goals"][max(0, i-window):i+1]) * 100 
                    for i in range(len(sae_history["goals"]))]
    ax4.plot(episodes_range, success_rate, label=f"Success Rate ({window} ep)")
    ax4.set_ylabel("Success Rate (%)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

# LOSS — 5º GRÁFICO (sanitiza qualquer tensor restante)
ax5 = axes[4]
sae_history["loss"] = [
    v.detach().cpu().item() if torch.is_tensor(v) else float(v)
    for v in sae_history["loss"]
]
ax5.plot(episodes_range, sae_history["loss"], label="Loss", alpha=0.7)
ax5.set_ylabel("Loss")
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.set_xlabel("Episode")

# Marcar branches
if sae_history["branch_changes"]:
    for bc in sae_history["branch_changes"]:
        ep = bc["episode"]
        for ax in axes:
            ax.axvline(ep, color="red", linestyle="--", alpha=0.5, linewidth=1)

        ymax = ax1.get_ylim()[1]
        ax1.text(ep + 2, ymax * 0.9, 
                 f"L{bc['layer']}\nvar={bc['var_ratio']:.3f}", 
                 rotation=90, color="red", fontsize=7, alpha=0.7,
                 verticalalignment='top')

plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig(f"plots/{base_name}_sae_collab_training.png", dpi=150)
print(f"[INFO] Gráfico salvo em plots/{base_name}_sae_collab_training.png")
plt.show()