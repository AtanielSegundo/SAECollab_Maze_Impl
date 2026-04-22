"""
WorldCist'26 -- Shared utilities for single-run and multi-run experiments.
"""

import os
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


def plot_aligned_to_first(baseline_history, sae_history, baseline_first, sae_first, base_name,
                          pre=10, post=80):
    """Create and save a comparison figure aligned to first-success (t=0)."""
    length = pre + post + 1
    x = np.arange(-pre, post + 1)

    # Rewards aligned
    b_rewards = [np.nan] * length
    s_rewards = [np.nan] * length
    if baseline_first is not None:
        b_slice = pad_to_length(baseline_history.get("rewards", []), baseline_first - 1 + post + 1)[max(0, baseline_first - 1 - pre): baseline_first - 1 - pre + length]
        b_rewards[:len(b_slice)] = b_slice
    if sae_first is not None:
        s_slice = pad_to_length(sae_history.get("rewards", []), sae_first - 1 + post + 1)[max(0, sae_first - 1 - pre): sae_first - 1 - pre + length]
        s_rewards[:len(s_slice)] = s_slice

    # Success rate aligned
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
# Environment setup
# --------------------
def setup_environment(file_path):
    """Setup maze environment according to paper specifications"""
    if not os.path.exists(file_path) or file_path == '':
        print(f"[ERROR] File Not Found: {file_path}")
        print("Usage: python3 worldcist.py <maze_file.maze>")
        exit(-1)

    base_name = os.path.basename(file_path).split(".")[0]
    raw_env = MazeEnv(file_path, rewards_scaled=False, pass_through_walls=False)

    print(f"\n{'='*80}")
    print(f"WORLDCIST'26 EXPERIMENTAL SETUP")
    print(f"{'='*80}")
    print(f"Environment: {base_name}")
    print(f"Grid Size: {raw_env.rows}x{raw_env.cols} ({raw_env.rows * raw_env.cols} cells)")
    print(f"Start Position: {raw_env.agent_start}")
    print(f"Goal Position: {raw_env.agent_goal}")
    print(f"Wall Cells: {raw_env.rows * raw_env.cols - raw_env.opens_count}")
    print(f"{'='*80}\n")

    train_state_encoder = StateEncoder.ONE_HOT
    env = MazeGymWrapper(
        raw_env, train_state_encoder,
        num_last_states=1,
        possible_actions_feature=True,
        visited_count=False
    )

    return env, base_name, raw_env


# --------------------
# Evaluation
# --------------------
def eval_agent_find_path(agent, env, max_steps):
    """Avalia se agente consegue alcancar goal deterministicamente"""
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


# --------------------
# Training loop
# --------------------
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

    print(f"\n{'='*80}")
    print(f"TRAINING {agent_name}")
    print(f"{'='*80}")
    print(f"{'Episode':>7} {'Steps':>6} {'Reward':>8} {'Goals':>6} {'SuccRate':>8} "
          f"{'Loss':>10} {'Eps':>6}", end="")
    if is_sae:
        print(f" {'Params':>10} {'Branches':>8}")
    else:
        print()
    print(f"{'-'*80}")

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
                    print(f"\n[INFO] PATH_TO_GOAL_LEARNED no episodio {ep}")
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
                    print(f"\n{'='*80}")
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
                    print(f"  Parameters: {current_params:,} -> {new_params:,} (+{new_params-current_params:,})")
                    print(f"{'='*80}\n")

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

    print(f"\n{'='*80}")
    print(f"{agent_name} TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"Total Episodes: {episodes}")
    print(f"Total Goals Reached: {goals_count}")
    print(f"Final Success Rate (last 20 ep): {sum(history['goals'][-20:])/20*100:.1f}%")
    print(f"Average Reward (last 20 ep): {np.mean(history['rewards'][-20:]):.3f}")
    if is_sae:
        print(f"Total Branches Added: {branches_added}")
        if len(history.get("parameters_over_time", [])) > 0:
            print(f"Final Parameters: {history['parameters_over_time'][-1]:,}")
    print(f"{'='*80}\n")

    return history
