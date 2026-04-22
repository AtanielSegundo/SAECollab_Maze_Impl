"""
WorldCist'26 -- Single-run baseline vs SAE comparison.
"""

import sys
import os
import json
from datetime import datetime

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from worldcist.common import (
    pad_to_length, rolling_mean_nan, first_success_episode, maybe_percent,
    plot_aligned_to_first, setup_environment, train_agent, eval_agent_find_path,
)
from models.QModels import SAECollabDDQN, TorchDDQN, exp_decay_factor_to
from models.AStar import AStarQModel, a_star_maze_solve
from models.Path import get_best_path


def create_baseline_agent(state_size, action_size, max_steps, episodes):
    """Create baseline DQN agent with architecture from paper (1500, 750)"""
    TOTAL_STEPS = max_steps * episodes

    policy_net = nn.Sequential(
        nn.Linear(state_size, 1500),
        nn.ReLU(),
        nn.Linear(1500, 750),
        nn.ReLU(),
        nn.Linear(750, action_size),
    )

    epsilon_decay = exp_decay_factor_to(
        final_epsilon=0.1,
        final_step=TOTAL_STEPS,
        epsilon_start=1.0,
        convergence_threshold=0.01
    )

    agent = TorchDDQN(
        sequential_list=policy_net,
        state_size=state_size,
        action_size=action_size,
        lr=1e-5,
        gamma=0.99,
        batch_size=1024,
        epsilon_start=1.0,
        learn_interval=4,
        epsilon_final=0.1,
        tau=0.005,
        grad_clip=10.0,
        min_replay_size=2048,
        epsilon_decay=epsilon_decay,
        target_update=int(TOTAL_STEPS / 50),
        buffer_size=100_000
    )

    total_params = sum(p.numel() for p in agent.policy_net.parameters())
    print(f"\n[BASELINE DQN]")
    print(f"  Architecture: {state_size} -> 1500 -> 750 -> {action_size}")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Learning Rate: 1e-5")
    print(f"  Replay Buffer: 100,000")

    return agent, total_params


def create_sae_collab_agent(state_size, action_size, max_steps, episodes, env):
    """Create SAE CollabNet agent with constructive architecture"""
    TOTAL_STEPS = max_steps * episodes
    INITIAL_HIDDEN = max(128, (env.rows * env.cols * action_size) // 4)
    NEW_BRANCH_HIDDEN = max(16, (env.rows * env.cols * action_size) // 4)
    EXTRA_HIDDEN = max(16, (env.rows * env.cols * action_size) // 4)

    epsilon_decay = exp_decay_factor_to(
        final_epsilon=0.1,
        final_step=TOTAL_STEPS,
        epsilon_start=1.0,
        convergence_threshold=0.01
    )

    agent = SAECollabDDQN(
        state_size=state_size,
        action_size=action_size,
        first_hidden_size=INITIAL_HIDDEN,
        hidden_activation=nn.ReLU(),
        out_activation=nn.Identity(),
        accelerate_etas=True,
        lr=[1e-5, 5e-5],
        gamma=0.99,
        batch_size=1024,
        buffer_size=100_000,
        epsilon_start=1.0,
        epsilon_final=0.1,
        epsilon_decay=epsilon_decay,
        target_update=int(TOTAL_STEPS / 50),
        learn_interval=4,
        tau=0.005,
        grad_clip=10.0,
        min_replay_size=2048,
    )

    total_params = sum(p.numel() for p in agent.policy_net.parameters())
    print(f"\n[SAE COLLABNET DQN]")
    print(f"  Initial Architecture: {state_size} -> {INITIAL_HIDDEN} -> {action_size}")
    print(f"  Initial Parameters: {total_params:,}")
    print(f"  Branch Hidden Size: {NEW_BRANCH_HIDDEN}")
    print(f"  Learning Rates: [1e-5, 5e-5]")
    print(f"  Max Branches: 4")

    return agent, INITIAL_HIDDEN, NEW_BRANCH_HIDDEN, EXTRA_HIDDEN


def plot_comparison(baseline_history, sae_history, base_name, episodes, baseline_params):
    """Generate publication-quality comparison plots (robustified)."""

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    episodes_len = int(episodes)
    baseline_history = baseline_history or {}
    sae_history = sae_history or {}
    for key in ["rewards", "losses", "goals", "steps", "success_rate_window", "cumulative_goals", "parameters_over_time"]:
        baseline_history.setdefault(key, [])
        sae_history.setdefault(key, [])

    b_rewards = pad_to_length(baseline_history["rewards"], episodes_len)
    s_rewards = pad_to_length(sae_history["rewards"], episodes_len)
    b_losses = pad_to_length(baseline_history["losses"], episodes_len)
    s_losses = pad_to_length(sae_history["losses"], episodes_len)
    b_success = maybe_percent(pad_to_length(baseline_history["success_rate_window"], episodes_len, 0))
    s_success = maybe_percent(pad_to_length(sae_history["success_rate_window"], episodes_len, 0))
    b_cumgoals = pad_to_length(baseline_history["cumulative_goals"], episodes_len, 0)
    s_cumgoals = pad_to_length(sae_history["cumulative_goals"], episodes_len, 0)
    b_steps = pad_to_length(baseline_history["steps"], episodes_len, np.nan)
    s_steps = pad_to_length(sae_history["steps"], episodes_len, np.nan)
    s_params = pad_to_length(sae_history.get("parameters_over_time", []), episodes_len, baseline_params)

    episodes_range = np.arange(1, episodes_len + 1)
    window = 20

    ax1 = fig.add_subplot(gs[0, 0])
    baseline_ma = rolling_mean_nan(b_rewards, window)
    sae_ma = rolling_mean_nan(s_rewards, window)
    ax1.plot(episodes_range, b_rewards, alpha=0.15, linewidth=0.6, color='tab:blue')
    ax1.plot(episodes_range, baseline_ma, label='Baseline DQN', color='tab:blue', linewidth=2)
    ax1.plot(episodes_range, s_rewards, alpha=0.15, linewidth=0.6, color='tab:orange')
    ax1.plot(episodes_range, sae_ma, label='SAE CollabNet', color='tab:orange', linewidth=2)
    ax1.set_xlabel('Episode', fontsize=11); ax1.set_ylabel('Cumulative Reward', fontsize=11)
    ax1.set_title('(a) Cumulative Reward per Episode', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(episodes_range, b_success, label='Baseline DQN', color='tab:blue', linewidth=2)
    ax2.plot(episodes_range, s_success, label='SAE CollabNet', color='tab:orange', linewidth=2)
    ax2.set_xlabel('Episode', fontsize=11); ax2.set_ylabel('Success Rate (%)', fontsize=11)
    ax2.set_title('(b) Success Rate (20-episode window)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3); ax2.set_ylim([0, 105])

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(episodes_range, b_cumgoals, label='Baseline DQN', color='tab:blue', linewidth=2)
    ax3.plot(episodes_range, s_cumgoals, label='SAE CollabNet', color='tab:orange', linewidth=2)
    ax3.set_xlabel('Episode', fontsize=11); ax3.set_ylabel('Cumulative Goals', fontsize=11)
    ax3.set_title('(c) Cumulative Goals Reached', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=10); ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(episodes_range, rolling_mean_nan(b_losses, window), label='Baseline DQN', color='tab:blue', linewidth=2)
    ax4.plot(episodes_range, rolling_mean_nan(s_losses, window), label='SAE CollabNet', color='tab:orange', linewidth=2)
    ax4.set_xlabel('Episode', fontsize=11); ax4.set_ylabel('TD Loss', fontsize=11)
    ax4.set_title('(d) Training Loss', fontsize=11, fontweight='bold')
    try: ax4.set_yscale('log')
    except Exception: pass
    ax4.legend(fontsize=10); ax4.grid(True, alpha=0.3, which='both')

    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(episodes_range, s_params, label='SAE CollabNet', color='tab:orange', linewidth=2)
    ax5.axhline(y=baseline_params, color='tab:blue', linestyle='--', linewidth=2, label='Baseline DQN (const)')
    ax5.set_xlabel('Episode', fontsize=11); ax5.set_ylabel('Number of Parameters', fontsize=11)
    ax5.set_title('(e) Network Parameters Evolution', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10); ax5.grid(True, alpha=0.3); ax5.ticklabel_format(style='plain', axis='y')

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(episodes_range, rolling_mean_nan(b_steps, window), label='Baseline DQN', color='tab:blue', linewidth=2)
    ax6.plot(episodes_range, rolling_mean_nan(s_steps, window), label='SAE CollabNet', color='tab:orange', linewidth=2)
    ax6.set_xlabel('Episode', fontsize=11); ax6.set_ylabel('Steps per Episode', fontsize=11)
    ax6.set_title('(f) Episode Length (efficiency)', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10); ax6.grid(True, alpha=0.3)

    # Mark first-success and branch insertions
    baseline_first = first_success_episode(baseline_history)
    sae_first = first_success_episode(sae_history)

    if baseline_first is not None:
        ax1.axvline(baseline_first, color='tab:blue', linestyle='--', linewidth=1.0, alpha=0.9)
        ax2.axvline(baseline_first, color='tab:blue', linestyle='--', linewidth=1.0, alpha=0.6)
    if sae_first is not None:
        ax1.axvline(sae_first, color='tab:orange', linestyle='--', linewidth=1.0, alpha=0.9)
        ax2.axvline(sae_first, color='tab:orange', linestyle='--', linewidth=1.0, alpha=0.6)

    for insertion in sae_history.get("branch_insertions", []):
        ep = insertion.get("episode", None)
        params = insertion.get("parameters", None)
        if ep is None: continue
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.axvline(x=ep, color='red', linestyle=':', alpha=0.5, linewidth=1)
        if params is not None:
            ax5.scatter(ep, params, color='red', s=100, zorder=5, marker='*', label='_nolegend_')

    os.makedirs("plots", exist_ok=True)
    filename = f"plots/{base_name}_worldcist_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[INFO] Comparison plot saved: {filename}")

    plot_aligned_to_first(baseline_history, sae_history, baseline_first, sae_first, base_name)
    return filename


def save_results(baseline_history, sae_history, baseline_path, sae_path,
                 a_star_path, base_name, baseline_params):
    """Save experimental results to JSON"""
    results = {
        "experiment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "environment": base_name,
        "baseline_dqn": {
            "final_success_rate": sum(baseline_history["goals"][-20:]) / 20 * 100 if len(baseline_history["goals"]) >= 20 else sum(baseline_history["goals"]) / max(1, len(baseline_history["goals"])) * 100,
            "final_avg_reward": float(np.mean(baseline_history["rewards"][-20:])) if len(baseline_history["rewards"]) >= 20 else float(np.mean(baseline_history["rewards"])),
            "total_goals": sum(baseline_history["goals"]),
            "path_length": baseline_path.len if baseline_path else 0,
            "similarity_to_astar": baseline_path.similarity_to(a_star_path) if baseline_path and baseline_path.len > 0 else 0,
        },
        "sae_collabnet": {
            "final_success_rate": sum(sae_history["goals"][-20:]) / 20 * 100 if len(sae_history["goals"]) >= 20 else sum(sae_history["goals"]) / max(1, len(sae_history["goals"])) * 100,
            "final_avg_reward": float(np.mean(sae_history["rewards"][-20:])) if len(sae_history["rewards"]) >= 20 else float(np.mean(sae_history["rewards"])),
            "total_goals": sum(sae_history["goals"]),
            "branches_added": len(sae_history.get("branch_insertions", [])),
            "final_parameters": sae_history.get("parameters_over_time", [-1])[-1] if len(sae_history.get("parameters_over_time", [])) > 0 else None,
            "path_length": sae_path.len if sae_path else 0,
            "similarity_to_astar": sae_path.similarity_to(a_star_path) if sae_path and sae_path.len > 0 else 0,
            "branch_insertions": sae_history.get("branch_insertions", [])
        },
        "optimal_astar": {"path_length": a_star_path.len}
    }

    os.makedirs("results", exist_ok=True)
    filename = f"results/{base_name}_worldcist_results.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Results saved: {filename}")


def main():
    """Main experimental workflow"""
    file_path = sys.argv[-1] if len(sys.argv) > 1 else ""
    env, base_name, raw_env = setup_environment(file_path)

    EPISODES = 200
    STATE_SIZE = env.state_size
    ACTION_SIZE = env.action_size
    MAX_STEPS = env.maze.opens_count * ACTION_SIZE

    print(f"[INFO] Experimental Parameters:")
    print(f"  Episodes: {EPISODES}")
    print(f"  Max Steps per Episode: {MAX_STEPS}")
    print(f"  State Size: {STATE_SIZE}")
    print(f"  Action Size: {ACTION_SIZE}")

    print(f"\n[INFO] Computing A* optimal solution...")
    a_star_path = a_star_maze_solve(raw_env)
    a_star_model = AStarQModel(env)
    print(f"[INFO] A* optimal path length: {a_star_path.len}")

    baseline_agent, baseline_params = create_baseline_agent(STATE_SIZE, ACTION_SIZE, MAX_STEPS, EPISODES)
    sae_agent, init_hidden, new_branch_hidden, extra_hidden = create_sae_collab_agent(
        STATE_SIZE, ACTION_SIZE, MAX_STEPS, EPISODES, env
    )
    sae_config = {"INITIAL_HIDDEN": init_hidden, "NEW_BRANCH_HIDDEN": new_branch_hidden, "EXTRA_HIDDEN": extra_hidden}

    baseline_history = train_agent(baseline_agent, env, EPISODES, MAX_STEPS, "BASELINE DQN", is_sae=False)
    sae_history = train_agent(sae_agent, env, EPISODES, MAX_STEPS, "SAE COLLABNET DQN", is_sae=True, sae_config=sae_config)

    print(f"\n[INFO] Generating solution paths...")
    baseline_path = get_best_path(env, baseline_agent, max_steps=MAX_STEPS)
    sae_path = get_best_path(env, sae_agent, max_steps=MAX_STEPS)

    print(f"\nBaseline Path: {baseline_path.__str__(env)}")
    print(f"SAE Path: {sae_path.__str__(env)}")
    print(f"A* Path: {a_star_path.__str__(env)}")

    os.makedirs("models", exist_ok=True)
    sae_agent.save(f"models/{base_name}_sae_worldcist.pth")
    print(f"\n[INFO] Models saved in models/")

    print(f"\n[INFO] Generating comparison plots...")
    plot_comparison(baseline_history, sae_history, base_name, EPISODES, baseline_params)

    save_results(baseline_history, sae_history, baseline_path, sae_path, a_star_path, base_name, baseline_params)

    print(f"\n[INFO] Experiment completed successfully!")
    print(f"[INFO] Check plots/ and results/ directories for outputs")
    plt.show()


if __name__ == "__main__":
    main()
