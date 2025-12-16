"""
WorldCist'26 Experimental Setup
Comparison of Baseline DQN vs SAE CollabNet DQN for Robot Navigation

Adapted: multi-run (N seeds) aggregation, mean ± std plotting, robust plotting fixes,
and correct branch insertion annotations.
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
from models.AStar import AStarQModel, a_star_maze_solve
from models.Path import Path, get_best_path
from models.QTable import genearate_qtable_from_model
from StackedCollab.collabNet import MutationMode


# --------------------
# Utility helpers
# --------------------
def pad_to_length(lst, length, pad_value=np.nan):
    if lst is None:
        return [pad_value] * length
    if len(lst) >= length:
        return list(lst[:length])
    return list(lst) + [pad_value] * (length - len(lst))


def rolling_mean_nan(arr, window):
    """Compute rolling mean (inclusive) at each index using window length, ignoring NaNs."""
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n == 0:
        return []
    out = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - window)
        win = arr[start:i + 1]
        if np.isnan(win).all():
            out[i] = np.nan
        else:
            out[i] = np.nanmean(win)
    return out.tolist()


def first_success_episode(history):
    """Return 1-indexed episode of first success or None."""
    goals = history.get("goals", [])
    for i, g in enumerate(goals):
        if g:
            return i + 1
    return None


def maybe_percent(series):
    """If series values appear in [0,1], convert to percents."""
    if series is None or len(series) == 0:
        return series
    arr = np.asarray(series, dtype=float)
    if np.isnan(arr).all():
        return series
    if np.nanmax(arr) <= 1.0:
        return (arr * 100.0).tolist()
    return series


def set_seed(seed):
    """Set seeds for reproducibility across numpy, random, torch."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# --------------------
# Single-run aligned plot (kept for debug)
# --------------------
def plot_aligned_to_first(baseline_history, sae_history, baseline_first, sae_first, base_name,
                          pre=10, post=80):
    """Create and save a comparison figure aligned to first-success (t=0)."""
    length = pre + post + 1
    x = np.arange(-pre, post + 1)

    # Rewards aligned
    b_rewards = [np.nan] * length
    s_rewards = [np.nan] * length
    if baseline_first is not None:
        b_slice = pad_to_length(baseline_history.get("rewards", []), baseline_first - 1 + post + 1)[
                  max(0, baseline_first - 1 - pre): baseline_first - 1 - pre + length]
        b_rewards[:len(b_slice)] = b_slice
    if sae_first is not None:
        s_slice = pad_to_length(sae_history.get("rewards", []), sae_first - 1 + post + 1)[
                  max(0, sae_first - 1 - pre): sae_first - 1 - pre + length]
        s_rewards[:len(s_slice)] = s_slice

    # Success rate aligned (prefer using windowed success rates if available)
    def aligned_success(history, first):
        arr = [np.nan] * length
        if first is None:
            return arr
        success_arr = history.get("success_rate_window", None)
        if success_arr and len(success_arr) > 0:
            padded = pad_to_length(success_arr, first - 1 + post + 1)
            slice_ = padded[max(0, first - 1 - pre): max(0, first - 1 - pre) + length]
            for i, v in enumerate(slice_):
                arr[i] = v
            return arr
        # fallback: smoothed goals -> percent
        goals = pad_to_length(history.get("goals", []), first - 1 + post + 1, 0)
        slice_goals = goals[max(0, first - 1 - pre): max(0, first - 1 - pre) + length]
        if len(slice_goals) == 0:
            return arr
        conv = np.convolve(slice_goals, np.ones(5) / 5.0, mode='same') * 100.0
        for i, v in enumerate(conv):
            arr[i] = v
        return arr

    b_succ = aligned_success(baseline_history, baseline_first)
    s_succ = aligned_success(sae_history, sae_first)

    fig2, (ra, sa) = plt.subplots(1, 2, figsize=(14, 5))
    ra.plot(x, b_rewards, label='Baseline DQN', linewidth=2, color='tab:blue')
    ra.plot(x, s_rewards, label='SAE CollabNet', linewidth=2, color='tab:orange')
    ra.axvline(0, color='k', linestyle='--', alpha=0.6)
    ra.set_title('Reward aligned to first-success (t=0)')
    ra.set_xlabel('Episodes relative to first success (negative = before)')
    ra.set_ylabel('Cumulative Reward')
    ra.grid(True, alpha=0.25)
    ra.legend()

    sa.plot(x, b_succ, label='Baseline DQN', linewidth=2, color='tab:blue')
    sa.plot(x, s_succ, label='SAE CollabNet', linewidth=2, color='tab:orange')
    sa.axvline(0, color='k', linestyle='--', alpha=0.6)
    sa.set_title('Success rate aligned to first-success (t=0)')
    sa.set_xlabel('Episodes relative to first success')
    sa.set_ylabel('Success Rate (%)')
    sa.set_ylim([0, 105])
    sa.grid(True, alpha=0.25)
    sa.legend()

    os.makedirs("plots", exist_ok=True)
    fname_aligned = f"plots/{base_name}_aligned_firstsuccess.png"
    plt.savefig(fname_aligned, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"[INFO] Aligned-first-success plot saved: {fname_aligned}")


# --------------------
# Environment & Agents (unchanged)
# --------------------
def setup_environment(file_path):
    """Setup maze environment according to paper specifications"""
    if not os.path.exists(file_path) or file_path == '':
        print(f"[ERROR] File Not Found: {file_path}")
        print("Usage: python3 worldcist.py <maze_file.maze>")
        exit(-1)

    base_name = os.path.basename(file_path).split(".")[0]
    raw_env = MazeEnv(file_path, rewards_scaled=False, pass_through_walls=False)

    print(f"\n{'=' * 80}")
    print(f"WORLDCIST'26 EXPERIMENTAL SETUP")
    print(f"{'=' * 80}")
    print(f"Environment: {base_name}")
    print(f"Grid Size: {raw_env.rows}x{raw_env.cols} ({raw_env.rows * raw_env.cols} cells)")
    print(f"Start Position: {raw_env.agent_start}")
    print(f"Goal Position: {raw_env.agent_goal}")
    print(f"Wall Cells: {raw_env.rows * raw_env.cols - raw_env.opens_count}")
    print(f"{'=' * 80}\n")

    train_state_encoder = StateEncoder.ONE_HOT
    env = MazeGymWrapper(
        raw_env, train_state_encoder,
        num_last_states=1,
        possible_actions_feature=True,
        visited_count=False
    )

    return env, base_name, raw_env


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
    print(f"  Architecture: {state_size} → 1500 → 750 → {action_size}")
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
    print(f"  Initial Architecture: {state_size} → {INITIAL_HIDDEN} → {action_size}")
    print(f"  Initial Parameters: {total_params:,}")
    print(f"  Branch Hidden Size: {NEW_BRANCH_HIDDEN}")
    print(f"  Learning Rates: [1e-5, 5e-5]")
    print(f"  Max Branches: 4")

    return agent, INITIAL_HIDDEN, NEW_BRANCH_HIDDEN, EXTRA_HIDDEN


def eval_agent_find_path(agent, env, max_steps):
    """Avalia se agente consegue alcançar goal deterministicamente"""
    old_epsilon = getattr(agent, "epsilon", None)
    prev_training = getattr(agent.policy_net, "training", True)
    agent.policy_net.eval()

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
                action = agent.act(state)
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


def train_agent(agent, env, episodes, max_steps, agent_name,
                is_sae=False, sae_config=None):
    """Train agent and collect metrics"""
    history = {
        "rewards": [],
        "losses": [],
        "goals": [],
        "steps": [],
        "epsilon": [],
        "cumulative_goals": [],
        "success_rate_window": [],
        "avg_reward_window": [],
    }

    if is_sae:
        history.update({
            "branch_insertions": [],
            "parameters_over_time": [],
            "variance_ratio": [],
        })

        branches_added = 0
        episodes_since_last_branch = 0
        PATIENCE = 15
        MIN_VARIANCE = 0.6
        MAX_BRANCHES = 4
        MIN_GOALS_BEFORE_BRANCH = 10
        TEST_GOAL_REACHED_INTERVAL = 5
        TEST_GOAL_REACHED = False

    print(f"\n{'=' * 80}")
    print(f"TRAINING {agent_name}")
    print(f"{'=' * 80}")
    print(f"{'Episode':>7} {'Steps':>6} {'Reward':>8} {'Goals':>6} {'SuccRate':>8} "
          f"{'Loss':>10} {'Eps':>6}", end="")
    if is_sae:
        print(f" {'Params':>10} {'Branches':>8}")
    else:
        print()
    print(f"{'-' * 80}")

    goals_count = 0

    for ep in range(1, episodes + 1):
        state = env.reset().reshape(1, -1)
        episode_reward = 0.0
        episode_steps = 0
        goal_reached = False

        for step in range(max_steps):
            action = agent.act(state, eval=False)
            next_state, reward, done, extras = env.step(action)

            if env.isGoal(extras["raw_ns"]):
                goal_reached = True
                goals_count += 1

            next_state = next_state.reshape(1, -1)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            episode_steps += 1

            if done:
                break

        if is_sae:
            agent.policy_net.step_all_etas()

        agent.update_epsilon()

        # Record metrics
        history["rewards"].append(episode_reward)
        # Convert loss to Python float, handling CUDA tensors
        if hasattr(agent, 'loss'):
            if isinstance(agent.loss, torch.Tensor):
                loss_val = float(agent.loss.cpu().detach())
            else:
                loss_val = float(agent.loss) if agent.loss is not None else 0.0
        else:
            loss_val = 0.0
        history["losses"].append(loss_val)
        history["goals"].append(1 if goal_reached else 0)
        history["steps"].append(episode_steps)
        history["epsilon"].append(agent.epsilon)
        history["cumulative_goals"].append(goals_count)

        # Calculate rolling metrics
        window = 20
        if ep >= window:
            recent_goals = sum(history["goals"][-window:])
            success_rate = (recent_goals / window) * 100
            avg_reward = np.mean(history["rewards"][-window:])
            history["success_rate_window"].append(success_rate)
            history["avg_reward_window"].append(avg_reward)
        else:
            history["success_rate_window"].append(0)
            history["avg_reward_window"].append(episode_reward)

        # SAE-specific: branch insertion logic
        if is_sae:
            episodes_since_last_branch += 1
            current_params = sum(p.numel() for p in agent.policy_net.parameters())
            history["parameters_over_time"].append(current_params)

            if (ep % TEST_GOAL_REACHED_INTERVAL == 0) and (not TEST_GOAL_REACHED):
                found = eval_agent_find_path(agent, env, max_steps)
                TEST_GOAL_REACHED = found
                if found:
                    TEST_GOAL_REACHED = True
                    sae_path = get_best_path(env, agent, max_steps=max_steps)
                    print(f"\n[INFO] ✓ PATH_TO_GOAL_LEARNED no episódio {ep}")
                    print(sae_path.__str__(env))

            if (ep >= PATIENCE and ep % PATIENCE == 0 and
                    branches_added < MAX_BRANCHES):

                window_rewards = history["rewards"][-PATIENCE:]
                window_goals = history["goals"][-PATIENCE:]
                window_mean = np.mean(window_rewards)
                window_var = np.var(window_rewards)
                goals_in_window = sum(window_goals)

                var_ratio = window_var / abs(window_mean) if abs(window_mean) > 1e-6 else float('inf')
                history["variance_ratio"].append(var_ratio)

                should_add_branch = (
                        var_ratio < MIN_VARIANCE and
                        (goals_in_window >= MIN_GOALS_BEFORE_BRANCH or branches_added == 0) and
                        episodes_since_last_branch >= PATIENCE
                )

                if should_add_branch:
                    print(f"\n{'=' * 80}")
                    print(f"[BRANCH INSERTION at Episode {ep}]")
                    print(f"  Variance/Mean: {var_ratio:.4f} < {MIN_VARIANCE}")
                    print(f"  Goals in window: {goals_in_window}/{PATIENCE}")

                    agent.add_layer(
                        layer_hidden_size=sae_config["NEW_BRANCH_HIDDEN"],
                        layer_extra_size=sae_config["EXTRA_HIDDEN"],
                        k=1.0,
                        mutation_mode=MutationMode.Hidden,
                        target_fn=nn.ReLU(),
                        eta=0.0,
                        eta_increment=1 / episodes,
                        hidden_activation=nn.ReLU(),
                        out_activation=nn.Identity(),
                        extra_activation=nn.ReLU(),
                    )

                    branches_added += 1
                    episodes_since_last_branch = 0
                    new_params = sum(p.numel() for p in agent.policy_net.parameters())

                    history["branch_insertions"].append({
                        "episode": ep,
                        "branch_number": branches_added,
                        "parameters": new_params,
                        "var_ratio": var_ratio,
                        "goals_in_window": goals_in_window
                    })

                    print(f"  Branch {branches_added} added")
                    print(f"  Parameters: {current_params:,} → {new_params:,} (+{new_params - current_params:,})")
                    print(f"{'=' * 80}\n")

        # Logging
        if ep % 10 == 0:
            recent_success = sum(history["goals"][-20:]) / min(20, ep) * 100 if ep >= 20 else 0
            current_loss = history["losses"][-1] if history["losses"] else 0.0
            print(f"{ep:7d} {episode_steps:6d} {episode_reward:8.3f} "
                  f"{goals_count:6d} {recent_success:7.1f}% "
                  f"{current_loss:10.3e} {agent.epsilon:6.3f}", end="")
            if is_sae:
                print(f" {current_params:10,} {branches_added:8d}")
            else:
                print()

    print(f"\n{'=' * 80}")
    print(f"{agent_name} TRAINING COMPLETED")
    print(f"{'=' * 80}")
    print(f"Total Episodes: {episodes}")
    print(f"Total Goals Reached: {goals_count}")
    print(f"Final Success Rate (last 20 ep): {sum(history['goals'][-20:]) / 20 * 100:.1f}%")
    print(f"Average Reward (last 20 ep): {np.mean(history['rewards'][-20:]):.3f}")
    if is_sae:
        print(f"Total Branches Added: {branches_added}")
        if len(history.get("parameters_over_time", [])) > 0:
            print(f"Final Parameters: {history['parameters_over_time'][-1]:,}")
    print(f"{'=' * 80}\n")

    return history


# --------------------
# Aggregation helpers for multi-run
# --------------------
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
    # branch insertion counts per episode
    branch_counts = np.zeros((len(hist_list), eps), dtype=float)
    for i, h in enumerate(hist_list):
        for ins in h.get("branch_insertions", []):
            ep = ins.get("episode", None)
            if ep is not None and 1 <= ep <= eps:
                branch_counts[i, ep - 1] += 1
    aggregated["branch_count_mean"] = np.nanmean(branch_counts, axis=0)
    aggregated["branch_count_std"] = np.nanstd(branch_counts, axis=0)
    # first success stats
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

    # (a) Rewards
    ax1 = fig.add_subplot(gs[0, 0])
    b_mean, b_std = agg_b["rewards_mean"], agg_b["rewards_std"]
    s_mean, s_std = agg_s["rewards_mean"], agg_s["rewards_std"]
    ax1.plot(x, b_mean, color='tab:blue', label='Baseline mean', linewidth=2)
    ax1.fill_between(x, b_mean - b_std, b_mean + b_std, color='tab:blue', alpha=0.2)
    ax1.plot(x, s_mean, color='tab:orange', label='SAE mean', linewidth=2)
    ax1.fill_between(x, s_mean - s_std, s_mean + s_std, color='tab:orange', alpha=0.2)
    ax1.set_title('(a) Cumulative Reward per Episode (mean ± std)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Cumulative Reward')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # (b) Success rate
    ax2 = fig.add_subplot(gs[0, 1])
    b_mean_s, b_std_s = agg_b["success_rate_window_mean"], agg_b["success_rate_window_std"]
    s_mean_s, s_std_s = agg_s["success_rate_window_mean"], agg_s["success_rate_window_std"]
    ax2.plot(x, b_mean_s, color='tab:blue', label='Baseline mean', linewidth=2)
    ax2.fill_between(x, b_mean_s - b_std_s, b_mean_s + b_std_s, color='tab:blue', alpha=0.2)
    ax2.plot(x, s_mean_s, color='tab:orange', label='SAE mean', linewidth=2)
    ax2.fill_between(x, s_mean_s - s_std_s, s_mean_s + s_std_s, color='tab:orange', alpha=0.2)
    ax2.set_title('(b) Success Rate (mean ± std)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # (c) cumulative goals
    ax3 = fig.add_subplot(gs[1, 0])
    b_mean_c = agg_b["cumulative_goals_mean"]
    b_std_c = agg_b["cumulative_goals_std"]
    s_mean_c = agg_s["cumulative_goals_mean"]
    s_std_c = agg_s["cumulative_goals_std"]
    ax3.plot(x, b_mean_c, color='tab:blue', linewidth=2, label='Baseline mean')
    ax3.fill_between(x, b_mean_c - b_std_c, b_mean_c + b_std_c, color='tab:blue', alpha=0.2)
    ax3.plot(x, s_mean_c, color='tab:orange', linewidth=2, label='SAE mean')
    ax3.fill_between(x, s_mean_c - s_std_c, s_mean_c + s_std_c, color='tab:orange', alpha=0.2)
    ax3.set_title('(c) Cumulative Goals (mean ± std)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Cumulative Goals')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # (d) training loss
    ax4 = fig.add_subplot(gs[1, 1])
    b_mean_l = agg_b["losses_mean"]
    b_std_l = agg_b["losses_std"]
    s_mean_l = agg_s["losses_mean"]
    s_std_l = agg_s["losses_std"]
    ax4.plot(x, b_mean_l, color='tab:blue', linewidth=2, label='Baseline mean')
    ax4.fill_between(x, b_mean_l - b_std_l, b_mean_l + b_std_l, color='tab:blue', alpha=0.2)
    ax4.plot(x, s_mean_l, color='tab:orange', linewidth=2, label='SAE mean')
    ax4.fill_between(x, s_mean_l - s_std_l, s_mean_l + s_std_l, color='tab:orange', alpha=0.2)
    ax4.set_title('(d) Training Loss (mean ± std)')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('TD Loss')
    try:
        ax4.set_yscale('log')
    except Exception:
        pass
    ax4.grid(True, alpha=0.3, which='both')
    ax4.legend()

    # (e) parameters (only SAE mean vs baseline const)
    ax5 = fig.add_subplot(gs[2, 0])
    s_params_mean = agg_s["parameters_over_time_mean"]
    s_params_std = agg_s["parameters_over_time_std"]
    ax5.plot(x, s_params_mean, color='tab:orange', linewidth=2, label='SAE params mean')
    ax5.fill_between(x, s_params_mean - s_params_std, s_params_mean + s_params_std, color='tab:orange', alpha=0.2)
    ax5.axhline(y=baseline_params, color='tab:blue', linestyle='--', linewidth=2, label='Baseline (const)')
    ax5.set_title('(e) Network Parameters Evolution (mean ± std)')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Number of Parameters')
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    # (f) steps per episode
    ax6 = fig.add_subplot(gs[2, 1])
    b_mean_steps = agg_b["steps_mean"]
    b_std_steps = agg_b["steps_std"]
    s_mean_steps = agg_s["steps_mean"]
    s_std_steps = agg_s["steps_std"]
    ax6.plot(x, b_mean_steps, color='tab:blue', linewidth=2, label='Baseline mean')
    ax6.fill_between(x, b_mean_steps - b_std_steps, b_mean_steps + b_std_steps, color='tab:blue', alpha=0.2)
    ax6.plot(x, s_mean_steps, color='tab:orange', linewidth=2, label='SAE mean')
    ax6.fill_between(x, s_mean_steps - s_std_steps, s_mean_steps + s_std_steps, color='tab:orange', alpha=0.2)
    ax6.set_title('(f) Episode Length (mean ± std)')
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Steps per Episode')
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    plt.suptitle('Baseline DQN vs SAE CollabNet DQN - Mean ± Std over runs', fontsize=14, fontweight='bold', y=0.995)
    os.makedirs("plots", exist_ok=True)
    filename = f"plots/{base_name}_worldcist_comparison_meanstd.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Aggregated comparison plot saved: {filename}")
    return filename


# --------------------
# Save single-run results (kept)
# --------------------
def save_results(baseline_history, sae_history, baseline_path, sae_path, a_star_path, base_name):
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
        "optimal_astar": {
            "path_length": a_star_path.len
        }
    }

    os.makedirs("results", exist_ok=True)
    filename = f"results/{base_name}_worldcist_results.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] Results saved: {filename}")


# --------------------
# Multi-run main
# --------------------
def main():
    """Main workflow with multiple runs to obtain mean ± std."""
    file_path = sys.argv[-1] if len(sys.argv) > 1 else ""
    env, base_name, raw_env = setup_environment(file_path)

    # Parameters
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

    # Compute A* once
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

        # recreate environment wrapper to avoid internal RNG/state carryover
        raw_env = MazeEnv(file_path, rewards_scaled=False, pass_through_walls=False)
        env = MazeGymWrapper(raw_env, StateEncoder.ONE_HOT, num_last_states=1,
                             possible_actions_feature=True, visited_count=False)

        # create fresh agents
        baseline_agent, baseline_params = create_baseline_agent(STATE_SIZE, ACTION_SIZE, MAX_STEPS, EPISODES)
        sae_agent, init_hidden, new_branch_hidden, extra_hidden = create_sae_collab_agent(
            STATE_SIZE, ACTION_SIZE, MAX_STEPS, EPISODES, env
        )
        sae_config = {"INITIAL_HIDDEN": init_hidden, "NEW_BRANCH_HIDDEN": new_branch_hidden, "EXTRA_HIDDEN": extra_hidden}

        # train baseline
        b_hist = train_agent(baseline_agent, env, EPISODES, MAX_STEPS, "BASELINE DQN", is_sae=False)
        baseline_histories.append(b_hist)

        # train sae
        sae_hist = train_agent(sae_agent, env, EPISODES, MAX_STEPS, "SAE COLLABNET DQN", is_sae=True, sae_config=sae_config)
        sae_histories.append(sae_hist)

        # optional: save per-run models
        os.makedirs("models", exist_ok=True)
        sae_agent.save(f"models/{base_name}_sae_seed{seed}.pth")
        # baseline_agent.save(f"models/{base_name}_baseline_seed{seed}.pth")  # uncomment if needed

    # aggregate
    agg_b = aggregate_histories(baseline_histories, EPISODES)
    agg_s = aggregate_histories(sae_histories, EPISODES)

    # Plot aggregated mean ± std
    plot_comparison_aggregate_with_two(agg_b, agg_s, base_name, EPISODES, baseline_params)

    # Save aggregated results summary
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