"""
WorldCist'26 -- Multi-run experimental setup with mean+-std aggregation.
"""

import os
import sys
import json
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from env.MazeEnv import *
from env.MazeWrapper import StateEncoder, MazeGymWrapper
from models.QModels import SAECollabDDQN, TorchDDQN, exp_decay_factor_to
from models.AStar import a_star_maze_solve
from models.Path import get_best_path
from StackedCollab.collabNet import MutationMode

from worldcist.common import (
    setup_environment,
    train_agent,
    first_success_episode,
)


def set_seed(seed):
    """Set seeds for reproducibility across numpy, random, torch."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


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
        target_update=int(3 / 50),
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


def aggregate_histories(hist_list, episodes):
    """
    hist_list: list of history dicts (one per run).
    returns dict of aggregated arrays (mean/std) per-episode for keys used in plotting.
    """
    eps = int(episodes)
    keys = ["rewards", "losses", "success_rate_window", "cumulative_goals", "steps", "parameters_over_time"]
    aggregated = {}
    for key in keys:
        mat = np.full((len(hist_list), eps), np.nan, dtype=float)
        for i, h in enumerate(hist_list):
            arr = h.get(key, [])
            for j in range(min(len(arr), eps)):
                val = arr[j]
                try:
                    mat[i, j] = float(val)
                except Exception:
                    mat[i, j] = np.nan
        aggregated[key + "_mean"] = np.nanmean(mat, axis=0)
        aggregated[key + "_std"] = np.nanstd(mat, axis=0)
    branch_counts = np.zeros((len(hist_list), eps), dtype=float)
    for i, h in enumerate(hist_list):
        for ins in h.get("branch_insertions", []):
            ep = ins.get("episode", None)
            if ep is not None and 1 <= ep <= eps:
                branch_counts[i, ep - 1] += 1
    aggregated["branch_count_mean"] = np.nanmean(branch_counts, axis=0)
    aggregated["branch_count_std"] = np.nanstd(branch_counts, axis=0)
    firsts = []
    for h in hist_list:
        f = first_success_episode(h)
        firsts.append(np.nan if f is None else float(f))
    aggregated["first_success_mean"] = np.nanmean(firsts)
    aggregated["first_success_median"] = np.nanmedian(firsts)
    aggregated["first_success_all"] = firsts
    return aggregated


def plot_comparison_aggregate_with_two(agg_b, agg_s, base_name, episodes, baseline_params):
    """Plot aggregated mean/std for baseline (agg_b) and sae (agg_s)."""
    eps = int(episodes)
    x = np.arange(1, eps + 1)
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    b_mean, b_std = agg_b["rewards_mean"], agg_b["rewards_std"]
    s_mean, s_std = agg_s["rewards_mean"], agg_s["rewards_std"]
    ax1.plot(x, b_mean, color='tab:blue', label='Baseline mean', linewidth=2)
    ax1.fill_between(x, b_mean - b_std, b_mean + b_std, color='tab:blue', alpha=0.2)
    ax1.plot(x, s_mean, color='tab:orange', label='SAE mean', linewidth=2)
    ax1.fill_between(x, s_mean - s_std, s_mean + s_std, color='tab:orange', alpha=0.2)
    ax1.set_title('(a) Cumulative Reward per Episode (mean +- std)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Cumulative Reward')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    b_mean_s, b_std_s = agg_b["success_rate_window_mean"], agg_b["success_rate_window_std"]
    s_mean_s, s_std_s = agg_s["success_rate_window_mean"], agg_s["success_rate_window_std"]
    ax2.plot(x, b_mean_s, color='tab:blue', label='Baseline mean', linewidth=2)
    ax2.fill_between(x, b_mean_s - b_std_s, b_mean_s + b_std_s, color='tab:blue', alpha=0.2)
    ax2.plot(x, s_mean_s, color='tab:orange', label='SAE mean', linewidth=2)
    ax2.fill_between(x, s_mean_s - s_std_s, s_mean_s + s_std_s, color='tab:orange', alpha=0.2)
    ax2.set_title('(b) Success Rate (mean +- std)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = fig.add_subplot(gs[1, 0])
    b_mean_c = agg_b["cumulative_goals_mean"]
    b_std_c = agg_b["cumulative_goals_std"]
    s_mean_c = agg_s["cumulative_goals_mean"]
    s_std_c = agg_s["cumulative_goals_std"]
    ax3.plot(x, b_mean_c, color='tab:blue', linewidth=2, label='Baseline mean')
    ax3.fill_between(x, b_mean_c - b_std_c, b_mean_c + b_std_c, color='tab:blue', alpha=0.2)
    ax3.plot(x, s_mean_c, color='tab:orange', linewidth=2, label='SAE mean')
    ax3.fill_between(x, s_mean_c - s_std_c, s_mean_c + s_std_c, color='tab:orange', alpha=0.2)
    ax3.set_title('(c) Cumulative Goals (mean +- std)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Cumulative Goals')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 1])
    b_mean_l = agg_b["losses_mean"]
    b_std_l = agg_b["losses_std"]
    s_mean_l = agg_s["losses_mean"]
    s_std_l = agg_s["losses_std"]
    ax4.plot(x, b_mean_l, color='tab:blue', linewidth=2, label='Baseline mean')
    ax4.fill_between(x, b_mean_l - b_std_l, b_mean_l + b_std_l, color='tab:blue', alpha=0.2)
    ax4.plot(x, s_mean_l, color='tab:orange', linewidth=2, label='SAE mean')
    ax4.fill_between(x, s_mean_l - s_std_l, s_mean_l + s_std_l, color='tab:orange', alpha=0.2)
    ax4.set_title('(d) Training Loss (mean +- std)')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('TD Loss')
    try:
        ax4.set_yscale('log')
    except Exception:
        pass
    ax4.grid(True, alpha=0.3, which='both')
    ax4.legend()

    ax5 = fig.add_subplot(gs[2, 0])
    s_params_mean = agg_s["parameters_over_time_mean"]
    s_params_std = agg_s["parameters_over_time_std"]
    ax5.plot(x, s_params_mean, color='tab:orange', linewidth=2, label='SAE params mean')
    ax5.fill_between(x, s_params_mean - s_params_std, s_params_mean + s_params_std, color='tab:orange', alpha=0.2)
    ax5.axhline(y=baseline_params, color='tab:blue', linestyle='--', linewidth=2, label='Baseline (const)')
    ax5.set_title('(e) Network Parameters Evolution (mean +- std)')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Number of Parameters')
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    ax6 = fig.add_subplot(gs[2, 1])
    b_mean_steps = agg_b["steps_mean"]
    b_std_steps = agg_b["steps_std"]
    s_mean_steps = agg_s["steps_mean"]
    s_std_steps = agg_s["steps_std"]
    ax6.plot(x, b_mean_steps, color='tab:blue', linewidth=2, label='Baseline mean')
    ax6.fill_between(x, b_mean_steps - b_std_steps, b_mean_steps + b_std_steps, color='tab:blue', alpha=0.2)
    ax6.plot(x, s_mean_steps, color='tab:orange', linewidth=2, label='SAE mean')
    ax6.fill_between(x, s_mean_steps - s_std_steps, s_mean_steps + s_std_steps, color='tab:orange', alpha=0.2)
    ax6.set_title('(f) Episode Length (mean +- std)')
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Steps per Episode')
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    plt.suptitle('Baseline DQN vs SAE CollabNet DQN - Mean +- Std over runs',
                 fontsize=14, fontweight='bold', y=0.995)
    os.makedirs("plots", exist_ok=True)
    filename = f"plots/{base_name}_worldcist_comparison_meanstd.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Aggregated comparison plot saved: {filename}")
    return filename


def main():
    """Main workflow with multiple runs to obtain mean +- std."""
    file_path = sys.argv[-1] if len(sys.argv) > 1 else ""
    env, base_name, raw_env = setup_environment(file_path)

    EPISODES = 200
    N_RUNS = 10
    STATE_SIZE = env.state_size
    ACTION_SIZE = env.action_size
    MAX_STEPS = env.maze.opens_count * ACTION_SIZE

    print(f"[INFO] Experimental Parameters:")
    print(f"  Episodes: {EPISODES}")
    print(f"  Runs: {N_RUNS}")
    print(f"  Max Steps per Episode: {MAX_STEPS}")
    print(f"  State Size: {STATE_SIZE}")
    print(f"  Action Size: {ACTION_SIZE}")

    print(f"\n[INFO] Computing A* optimal solution...")
    a_star_path = a_star_maze_solve(raw_env)
    print(f"[INFO] A* optimal path length: {a_star_path.len}")

    baseline_histories = []
    sae_histories = []

    base_seed = 1000
    baseline_params = None

    for run in range(N_RUNS):
        seed = base_seed + run
        print(f"\n[INFO] Starting run {run + 1}/{N_RUNS} (seed={seed})")
        set_seed(seed)

        raw_env = MazeEnv(file_path, rewards_scaled=False, pass_through_walls=False)
        env = MazeGymWrapper(raw_env, StateEncoder.ONE_HOT, num_last_states=1,
                             possible_actions_feature=True, visited_count=False)

        baseline_agent, baseline_params = create_baseline_agent(STATE_SIZE, ACTION_SIZE, MAX_STEPS, EPISODES)
        sae_agent, init_hidden, new_branch_hidden, extra_hidden = create_sae_collab_agent(
            STATE_SIZE, ACTION_SIZE, MAX_STEPS, EPISODES, env
        )
        sae_config = {"INITIAL_HIDDEN": init_hidden, "NEW_BRANCH_HIDDEN": new_branch_hidden,
                      "EXTRA_HIDDEN": extra_hidden}

        b_hist = train_agent(baseline_agent, env, EPISODES, MAX_STEPS, "BASELINE DQN", is_sae=False)
        baseline_histories.append(b_hist)

        sae_hist = train_agent(sae_agent, env, EPISODES, MAX_STEPS, "SAE COLLABNET DQN",
                               is_sae=True, sae_config=sae_config)
        sae_histories.append(sae_hist)

        os.makedirs("models", exist_ok=True)
        sae_agent.save(f"models/{base_name}_sae_seed{seed}.pth")

    agg_b = aggregate_histories(baseline_histories, EPISODES)
    agg_s = aggregate_histories(sae_histories, EPISODES)

    plot_comparison_aggregate_with_two(agg_b, agg_s, base_name, EPISODES, baseline_params)

    results = {
        "experiment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "environment": base_name,
        "runs": N_RUNS,
        "baseline": {
            "first_success_mean": float(np.nan if np.isnan(agg_b.get("first_success_mean")) else agg_b.get("first_success_mean")),
            "first_success_median": float(np.nan if np.isnan(agg_b.get("first_success_median")) else agg_b.get("first_success_median")),
            "final_success_rate_mean": float(np.nanmean([np.sum(h['goals'][-20:]) / 20 * 100 if len(h['goals']) >= 20 else np.sum(h['goals']) / max(1, len(h['goals'])) * 100 for h in baseline_histories])),
        },
        "sae": {
            "first_success_mean": float(np.nan if np.isnan(agg_s.get("first_success_mean")) else agg_s.get("first_success_mean")),
            "first_success_median": float(np.nan if np.isnan(agg_s.get("first_success_median")) else agg_s.get("first_success_median")),
            "final_success_rate_mean": float(np.nanmean([np.sum(h['goals'][-20:]) / 20 * 100 if len(h['goals']) >= 20 else np.sum(h['goals']) / max(1, len(h['goals'])) * 100 for h in sae_histories])),
        },
        "a_star": {"path_length": a_star_path.len}
    }
    os.makedirs("results", exist_ok=True)
    fname = f"results/{base_name}_aggregated_results.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Aggregated results saved: {fname}")

    print("[INFO] Multi-run experiment finished.")


if __name__ == "__main__":
    main()
