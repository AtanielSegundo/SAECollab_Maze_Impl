#!python3 ablation/training.py

import os
import time
import json
from typing import *
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from ablation.config import (
    GlobalHyperparameters, ModelArch, LayerInsertionType,
    LayerModeType, StateRepresentation,
)
from ablation.metrics import ModelTrainMetrics, TrainTargetClosure
from ablation.state import AblationProgramState

from models.QModels import (
    exp_decay_factor_to, TorchDDQN,
    SAECollabDDQN, ReservedSAECollabDDQN, NewLayerCfg,
)
from env.MazeEnv import MazeEnv
from env.MazeWrapper import StateEncoder, MazeGymWrapper
from env.GPUMazeWrapper import GPUMazeWrapper
from env.CPUMazeWrapper import CPUMazeWrapperAdapter
from StackedCollab.collabNet import LayersConfig, MutationMode


def gen_concrete_arch(
    base_width    : int,
    env           : MazeGymWrapper,
    architecute   : ModelArch,
    insertion_type: LayerInsertionType,
    min_width     : int = 64,
    max_width     : int = 2048,
) -> Optional[List[LayersConfig]]:
    
    base_width = 1 if architecute.is_static else max(min_width, min(max_width, base_width)) 
    
    if insertion_type is LayerInsertionType.CNT:
        cnt_hidden = architecute.width_multipliers.hidden * base_width
        cnt_extra  = architecute.width_multipliers.extra  * base_width
        cnt_hidden = np.ceil(cnt_hidden)
        cnt_extra  = np.ceil(cnt_extra)
        concrete_arch = [
            LayersConfig(
                cnt_hidden,
                env.action_size,
                cnt_extra
            )
            for _ in range(architecute.max_layers)
        ]
        return concrete_arch

    elif insertion_type is LayerInsertionType.CRT:
        concrete_arch = list()
        crt_hidden   = architecute.width_multipliers.hidden * base_width
        crt_extra    = architecute.width_multipliers.extra  * base_width
        delta_hidden = architecute.delta_width_multipliers.hidden * crt_hidden
        delta_extra  = architecute.delta_width_multipliers.extra * crt_extra
        for i in range(architecute.max_layers):
            hidden_size = max(1, np.ceil(crt_hidden + i * delta_hidden))
            extra_size  = max(0, np.ceil(crt_extra + i * delta_extra))
            concrete_arch.append(
                LayersConfig(hidden_size,
                             env.action_size,
                             extra_size)
            )
        return concrete_arch

    elif insertion_type is LayerInsertionType.DRT:
        concrete_arch = list()
        drt_hidden   = architecute.width_multipliers.hidden * base_width
        drt_extra    = architecute.width_multipliers.extra  * base_width
        delta_hidden = architecute.delta_width_multipliers.hidden * drt_hidden
        delta_extra  = architecute.delta_width_multipliers.extra * drt_extra
        for i in range(architecute.max_layers):
            hidden_size = max(1, np.ceil(drt_hidden - i * delta_hidden))
            extra_size  = max(0, np.ceil(drt_extra - i * delta_extra))
            concrete_arch.append(
                LayersConfig(hidden_size,
                             env.action_size,
                             extra_size)
            )
        return concrete_arch

    elif insertion_type is LayerInsertionType.ALT:
        concrete_arch = list()
        alt_hidden   = architecute.width_multipliers.hidden * base_width
        alt_extra    = architecute.width_multipliers.extra  * base_width
        delta_hidden = architecute.delta_width_multipliers.hidden * alt_hidden
        delta_extra  = architecute.delta_width_multipliers.extra * alt_extra
        sigma = 0.0
        for i in range(architecute.max_layers):
            if i > 0 : sigma = (2.0*np.random.random() - 1.0)
            alt_hidden += np.ceil(sigma * delta_hidden)
            alt_extra  += np.ceil(sigma * delta_extra)
            alt_hidden = max(1, alt_hidden)
            alt_extra  = max(0, alt_extra)
            concrete_arch.append(
                LayersConfig(alt_hidden,
                             env.action_size,
                             alt_extra)
            )
        return concrete_arch

    else:
        print(f"[ERROR] Insertion Type Invalid: {insertion_type}")
        return None

def save_concrete_arch_info(save_dir:str,concrete_arch:List[LayersConfig]):
    concrete_arch_repr = {"arch": [
        {
            "hidden":int(layer.hidden),
            "out"   :int(layer.out),
            "extra" :int(layer.extra)
        } for layer in concrete_arch
    ]}
    with open(os.path.join(save_dir,"concrete_arch.json"),"w") as f:
        json.dump(concrete_arch_repr,f,indent=4)

def eval_agent_deterministic(agent, env, max_steps):
    old_epsilon   = getattr(agent, "epsilon", None)
    prev_training = getattr(agent.policy_net, "training", True)
    agent.policy_net.eval()

    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0

    state = env.reset()
    # GPUMazeWrapper.reset() returns (1, state_size) GPU tensor
    # MazeGymWrapper.reset()  returns (state_size,) numpy  -- kept for compat
    if not isinstance(state, torch.Tensor):
        state = torch.from_numpy(
            np.asarray(state, dtype=np.float32)
        ).unsqueeze(0).to(agent.device)

    reached = False
    with torch.no_grad():
        for _ in range(max_steps):
            action                   = agent.act(state, eval=True)
            next_state, _, done, extras = env.step(action)
            if env.isGoal(extras.get("raw_ns", extras)):
                reached = True
                break
            if not isinstance(next_state, torch.Tensor):
                next_state = torch.from_numpy(
                    np.asarray(next_state, dtype=np.float32)
                ).unsqueeze(0).to(agent.device)
            state = next_state
            if done:
                break

    if hasattr(agent, "epsilon") and old_epsilon is not None:
        agent.epsilon = old_epsilon
    if prev_training:
        agent.policy_net.train()
    return reached

def train_saecollab_tolerance_model(
    save_path:str,
    env: MazeGymWrapper,
    model_arch:ModelArch,
    concrete_layer_arch: List[LayersConfig],
    hp:GlobalHyperparameters,
    mode_type:LayerModeType,
    mutation_mode:MutationMode,
    runs:int,
    verbose:bool = True,
    early_stop_episodes:int = 0
):
    if os.path.exists(save_path):
        return None
    agent = SAECollabDDQN(
        state_size=env.state_size,
        action_size=env.action_size,
        first_hidden_size=int(concrete_layer_arch[0].hidden),
        hidden_activation=model_arch.activation.hidden(),
        out_activation=model_arch.activation.out(),
        accelerate_etas=True,
        lr=[hp.learning_rate,hp.new_layer_learning_rate],
        gamma=hp.discount_factor,
        batch_size=hp.batch_size,
        epsilon_start=1.0,
        epsilon_final=0.1,
        epsilon_decay=hp.epsilon_decay,
        learn_interval=hp.steps_learn_interval,
        min_replay_size=max(1000,2*hp.batch_size),
        use_bias=model_arch.use_bias,
        use_per=getattr(hp, "use_per", False),
        per_alpha=getattr(hp, "per_alpha", 0.6),
        per_beta_start=getattr(hp, "per_beta_start", 0.4),
        per_beta_final=getattr(hp, "per_beta_final", 1.0),
        per_beta_steps=getattr(hp, "per_beta_steps", 100_000),
        per_eps=getattr(hp, "per_eps", 1e-6),
    )
    #agent.compile()
    parameters_cnt = sum(p.numel() for p in agent.policy_net.parameters())
    agent_metrics  = ModelTrainMetrics()
    cum_goals      = 0
    goal_reached   = np.zeros((hp.episodes),dtype=bool)
    current_branch = 0
    deterministic_reached = False
    episodes_since_last_branch = 0
    # Check if the goal at least once was reached
    goal_once_reached = False
    window_goal_count = 0
    consecutive_perfect = 0

    for episode in range(hp.episodes):
        epoch_start_time = time.perf_counter()
        cum_reward = 0.0
        cum_steps  = 0
        truncated  = False

        state_gpu = env.reset()               # (1, state_size) on GPU -- zero transfer
        state_np  = env.last_state_np         # (state_size,)   on CPU -- zero GPU sync

        for step in range(hp.max_steps):
            action                          = agent.act(state_gpu, eval=False)
            next_gpu, reward, done, extras  = env.step(action)

            if env.isGoal(extras["raw_ns"]):
                goal_reached[episode] = True
                goal_once_reached     = True
                cum_goals            += 1

            next_np = env.last_state_np       # CPU -- zero GPU sync
            # `done` here means TERMINATED (goal reached). When the loop exits
            # by max_steps without `done`, the episode is TRUNCATED: the last
            # transition is stored with done=False so the bootstrap target
            # `r + gamma * max Q(s',a')` continues to estimate the value of s'
            # (Gymnasium convention; avoids underestimating Q at the cap).
            agent.remember(state_np, action, reward, next_np, done)

            state_gpu = next_gpu              # GPU tensor -- zero transfer
            state_np  = next_np               # reuse; no recomputation next iter
            cum_reward += reward
            cum_steps   = step
            if done:
                break
        else:
            truncated = True

        agent.policy_net.step_all_etas()
        agent.update_epsilon()
        loss_val = float(agent.loss)

        # Incremental rolling window success rate -- O(1) per episode
        window_goal_count += int(goal_reached[episode])
        success_rate = 0.0
        if episode >= hp.rolling_window_size:
            window_goal_count -= int(goal_reached[episode - hp.rolling_window_size])
            success_rate = (window_goal_count / hp.rolling_window_size) * 100

        # New Branch Adding Logic
        episodes_since_last_branch += 1
        if (episode >= hp.insert_patience and
           episode % hp.insert_patience == 0 and
           current_branch < model_arch.max_layers):

           if deterministic_reached or (_dr:=eval_agent_deterministic(agent,env,hp.max_steps)):
                if not deterministic_reached and _dr:
                    if verbose:
                        print("[INFO] Deterministic Reached")
                deterministic_reached = deterministic_reached or _dr
           else :
               w_r         = agent_metrics.reward[-hp.insert_patience:]
               w_goals     = agent_metrics.cumulative_goals[-hp.insert_patience:]
               window_mean = np.mean(w_r)
               window_var  = np.var(w_r)
               goals_in_window = sum(w_goals)

               if abs(window_mean) > 1e-6:
                    var_ratio = window_var / abs(window_mean)
               else:
                    var_ratio = float('inf')

               should_add_branch = (
                    var_ratio < hp.insert_min_variance and
                    (goals_in_window >= hp.insert_min_goals
                     or (not goal_once_reached  or current_branch == 0)
                    ) and episodes_since_last_branch >= hp.insert_patience
               )

               if should_add_branch:
                   hidden_size = int(concrete_layer_arch[current_branch].hidden)
                   extra_size  = int(concrete_layer_arch[current_branch].extra if
                                     mode_type.value.use_extra_branch else 0)
                   extra_size  = None if extra_size == 0 else extra_size
                   agent.add_layer(
                       layer_hidden_size=hidden_size,
                       layer_extra_size=extra_size,
                       mutation_mode=mutation_mode,
                       target_fn=model_arch.activation.hidden(),
                       k=1.0,
                       eta=0.0,
                       eta_increment=1 / hp.episodes,
                       hidden_activation=model_arch.activation.hidden(),
                       out_activation=model_arch.activation.out(),
                       extra_activation=model_arch.activation.extra(),
                       is_k_trainable= mode_type.value.is_k_trainable,
                       use_bias=model_arch.use_bias
                   )
                   current_branch += 1
                   parameters_cnt = sum(p.numel() for p in agent.policy_net.parameters())
                   episodes_since_last_branch = 0
        # End
        delta_time = time.perf_counter() - epoch_start_time
        agent_metrics.append(episode,cum_reward,cum_goals,success_rate,
                             loss_val,cum_steps,parameters_cnt,
                             delta_time,current_branch
                             )

        # Early stopping: sustained 100% success rate
        if early_stop_episodes > 0:
            if success_rate >= 100.0:
                consecutive_perfect += 1
                if consecutive_perfect >= early_stop_episodes:
                    if verbose:
                        print(f"[INFO] Early stop at episode {episode} (100% for {early_stop_episodes} ep)")
                    break
            else:
                consecutive_perfect = 0

    agent.save(save_path)

    return agent_metrics


def train_reserved_saecollab_tolerance_model(
    save_path:str,
    env: MazeGymWrapper,
    model_arch:ModelArch,
    concrete_layer_arch: List[LayersConfig],
    hp:GlobalHyperparameters,
    mode_type:LayerModeType,
    mutation_mode:MutationMode,
    runs:int,
    verbose:bool = True,
    early_stop_episodes:int = 0
):
    if os.path.exists(save_path):
        return None

    # Pre-build all layer configs upfront
    eta_increment = 1 / hp.episodes
    reserved_layers_cfg = []

    # First layer (base)
    reserved_layers_cfg.append(NewLayerCfg(
        hidden_dim        = int(concrete_layer_arch[0].hidden),
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
        use_bias          = model_arch.use_bias
    ))

    # Remaining layers (reserved but frozen)
    for i in range(1, model_arch.max_layers):
        extra_dim = int(concrete_layer_arch[i].extra) if mode_type.value.use_extra_branch else None
        extra_dim = None if (extra_dim == 0) else extra_dim
        reserved_layers_cfg.append(NewLayerCfg(
            hidden_dim        = int(concrete_layer_arch[i].hidden),
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
            use_bias          = model_arch.use_bias
        ))

    agent = ReservedSAECollabDDQN(
        state_size            = env.state_size,
        action_size           = env.action_size,
        reserved_layers_cfg   = reserved_layers_cfg,
        accelerate_etas       = True,
        accelerate_factor     = 2.0,
        lr                    = [hp.learning_rate, hp.new_layer_learning_rate],
        gamma                 = hp.discount_factor,
        batch_size            = hp.batch_size,
        epsilon_start         = 1.0,
        epsilon_final         = 0.1,
        epsilon_decay         = hp.epsilon_decay,
        learn_interval        = hp.steps_learn_interval,
        min_replay_size       = max(1000, 2 * hp.batch_size),
        use_bias              = model_arch.use_bias,
        use_per               = getattr(hp, "use_per", False),
        per_alpha             = getattr(hp, "per_alpha", 0.6),
        per_beta_start        = getattr(hp, "per_beta_start", 0.4),
        per_beta_final        = getattr(hp, "per_beta_final", 1.0),
        per_beta_steps        = getattr(hp, "per_beta_steps", 100_000),
        per_eps               = getattr(hp, "per_eps", 1e-6),
    )
    #agent.compile()
    parameters_cnt = sum(
        p.numel()
        for layer in agent.policy_net.layers[:agent.policy_net.active_head + 1]
        for p in layer.parameters()
    )
    agent_metrics              = ModelTrainMetrics()
    cum_goals                  = 0
    goal_reached               = np.zeros((hp.episodes), dtype=bool)
    current_branch             = 0
    deterministic_reached      = False
    episodes_since_last_branch = 0
    goal_once_reached          = False
    window_goal_count          = 0
    consecutive_perfect        = 0

    for episode in range(hp.episodes):
        epoch_start_time = time.perf_counter()
        cum_reward = 0.0
        cum_steps  = 0
        truncated  = False

        state_gpu = env.reset()               # (1, state_size) on GPU -- zero transfer
        state_np  = env.last_state_np         # (state_size,)   on CPU -- zero GPU sync

        for step in range(hp.max_steps):
            action                          = agent.act(state_gpu, eval=False)
            next_gpu, reward, done, extras  = env.step(action)

            if env.isGoal(extras["raw_ns"]):
                goal_reached[episode] = True
                goal_once_reached     = True
                cum_goals             += 1

            next_np = env.last_state_np       # CPU -- zero GPU sync
            # Same semantics as the non-reserved variant: done=True only on
            # goal (terminated); on max_steps cap, episode is truncated and
            # the last transition keeps done=False so bootstrap continues.
            agent.remember(state_np, action, reward, next_np, done)

            state_gpu = next_gpu              # GPU tensor -- zero transfer
            state_np  = next_np               # reuse; no recomputation next iter
            cum_reward += reward
            cum_steps   = step
            if done:
                break
        else:
            truncated = True

        agent.policy_net.step_all_etas()
        agent.update_epsilon()
        loss_val = float(agent.loss)

        # Incremental rolling window success rate -- O(1) per episode
        window_goal_count += int(goal_reached[episode])
        success_rate = 0.0
        if episode >= hp.rolling_window_size:
            window_goal_count -= int(goal_reached[episode - hp.rolling_window_size])
            success_rate = (window_goal_count / hp.rolling_window_size) * 100

        # New Branch Logic -- same tolerance criteria, but uses use_next_layer()
        episodes_since_last_branch += 1
        max_branches = model_arch.max_layers - 1  # first layer already active

        if (episode >= hp.insert_patience and
            episode % hp.insert_patience == 0 and
            current_branch < max_branches):

            if deterministic_reached or (_dr := eval_agent_deterministic(agent, env, hp.max_steps)):
                if not deterministic_reached and _dr:
                    if verbose:
                        print("[INFO] Deterministic Reached")
                deterministic_reached = deterministic_reached or _dr
            else:
                w_r             = agent_metrics.reward[-hp.insert_patience:]
                w_goals         = agent_metrics.cumulative_goals[-hp.insert_patience:]
                window_mean     = np.mean(w_r)
                window_var      = np.var(w_r)
                goals_in_window = sum(w_goals)

                var_ratio = window_var / abs(window_mean) if abs(window_mean) > 1e-6 else float('inf')

                should_advance = (
                    var_ratio < hp.insert_min_variance and
                    (goals_in_window >= hp.insert_min_goals
                     or (not goal_once_reached or current_branch == 0)
                    ) and episodes_since_last_branch >= hp.insert_patience
                )

                if should_advance:
                    agent.use_next_layer()
                    current_branch += 1
                    parameters_cnt = sum(
                        p.numel()
                        for layer in agent.policy_net.layers[:agent.policy_net.active_head + 1]
                        for p in layer.parameters()
                    )
                    episodes_since_last_branch = 0

        delta_time = time.perf_counter() - epoch_start_time

        # print(episode, cum_reward, cum_goals, success_rate, loss_val, cum_steps, parameters_cnt, delta_time, current_branch)

        agent_metrics.append(episode, cum_reward, cum_goals, success_rate,
                             loss_val, cum_steps, parameters_cnt,
                             delta_time, current_branch)

        # Early stopping: sustained 100% success rate
        if early_stop_episodes > 0:
            if success_rate >= 100.0:
                consecutive_perfect += 1
                if consecutive_perfect >= early_stop_episodes:
                    if verbose:
                        print(f"[INFO] Early stop at episode {episode} (100% for {early_stop_episodes} ep)")
                    break
            else:
                consecutive_perfect = 0

    agent.save(save_path)
    return agent_metrics

def train_saecollab_spaced_model(
    save_path:str,
    env: MazeGymWrapper,
    model_arch:ModelArch,
    concrete_layer_arch: List[LayersConfig],
    hp:GlobalHyperparameters,
    mode_type:LayerModeType,
    mutation_mode:MutationMode,
    runs:int
):
    return None
    if os.path.exists(save_path):
        return None
    agent = SAECollabDDQN(
        state_size=env.state_size,
        action_size=env.action_size,
        first_hidden_size=int(concrete_layer_arch[0].hidden),
        hidden_activation=model_arch.activation.hidden(),
        out_activation=model_arch.activation.out(),
        accelerate_etas=True,
        lr=[hp.learning_rate,hp.new_layer_learning_rate],
        gamma=hp.discount_factor,
        batch_size=hp.batch_size,
        epsilon_start=1.0,
        epsilon_final=0.1,
        epsilon_decay=hp.epsilon_decay,
        learn_interval=hp.steps_learn_interval,
        min_replay_size=max(1000,2*hp.batch_size),
        use_bias=model_arch.use_bias
    )

    parameters_cnt = sum(p.numel() for p in agent.policy_net.parameters())
    agent_metrics  = ModelTrainMetrics()
    cum_goals      = 0
    goal_reached   = np.zeros((hp.episodes),dtype=bool)
    current_branch = 0
    branch_insertion_mod = hp.episodes // (model_arch.max_layers+1)
    deterministic_reached = False

    for episode in range(hp.episodes):
        cum_reward = 0.0
        cum_steps  = 0
        state = env.reset()
        state = state.reshape(1,-1)

        for step in range(hp.max_steps):
            action = agent.act(state, eval=False)
            next_state, reward, done, extras_dict = env.step(action)
            if env.isGoal(extras_dict["raw_ns"]):
                goal_reached[episode] = True
                cum_goals += 1
            next_state = next_state.reshape(1,-1)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            cum_reward += reward
            cum_steps = step
            if done:
                break

        agent.policy_net.step_all_etas()
        agent.update_epsilon()

        # Convert loss to Python float, handling CUDA tensors
        if hasattr(agent, 'loss'):
            if isinstance(agent.loss, torch.Tensor):
                loss_val = agent.loss.item()
            else:
                loss_val = float(agent.loss) if agent.loss is not None else 0.0
        else:
            loss_val = 0.0

        # Succes Rate Calculation
        success_rate = 0.0
        if episode >= hp.rolling_window_size:
            start_idx = max(0, episode - hp.rolling_window_size + 1)
            recent_window = goal_reached[start_idx: episode + 1]
            recent_goals = int(np.sum(recent_window))
            success_rate = (recent_goals / hp.rolling_window_size) * 100

        # New Branch Adding Logic

        if (episode+1) % branch_insertion_mod == 0 and current_branch < model_arch.max_layers:
            if deterministic_reached or (_dr:=eval_agent_deterministic(agent,env,hp.max_steps)):
                if not deterministic_reached and _dr:
                    print("[INFO] Deterministic Reached")
                deterministic_reached = deterministic_reached or _dr
            else:
                hidden_size = int(concrete_layer_arch[current_branch].hidden)
                extra_size  = int(concrete_layer_arch[current_branch].extra if mode_type.value.use_extra_branch else 0)
                extra_size  = None if extra_size == 0 else extra_size
                agent.add_layer(
                    layer_hidden_size=hidden_size,
                    layer_extra_size=extra_size,
                    mutation_mode=mutation_mode,
                    target_fn=model_arch.activation.hidden(),
                    k=1.0,
                    eta=0.0,
                    eta_increment=1 / branch_insertion_mod,
                    hidden_activation=model_arch.activation.hidden(),
                    out_activation=model_arch.activation.out(),
                    extra_activation=model_arch.activation.extra(),
                    is_k_trainable= mode_type.value.is_k_trainable,
                    use_bias=model_arch.use_bias
                )
                current_branch += 1
                parameters_cnt = sum(p.numel() for p in agent.policy_net.parameters())
        # End

        agent_metrics.append(episode,cum_reward,cum_goals,success_rate,loss_val,cum_steps,parameters_cnt)

        #if episode % 50 == 0:
        #    agent_metrics.pretty_print(5)

    agent.save(save_path)

    return agent_metrics

def train_baseline_dense_model(
    save_path:str,
    env: MazeGymWrapper,
    model_arch:ModelArch,
    concrete_layer_arch: List[LayersConfig],
    hp:GlobalHyperparameters,
    mode_type:LayerModeType,
    mutation_mode:MutationMode,
    runs:int,
    verbose:bool = True,
    early_stop_episodes:int = 0
):
    if os.path.exists(save_path):
        return None
    layers = []
    last_width = None
    for i in range(model_arch.max_layers):
        if i == 0 :
            width = int(concrete_layer_arch[i].hidden)
            layers.append(nn.Linear(env.state_size,width,bias=model_arch.use_bias.hidden))
            layers.append(model_arch.activation.hidden())
            last_width = width
        elif i == model_arch.max_layers - 1:
            layers.append(nn.Linear(last_width,env.action_size,bias=model_arch.use_bias.out))
            layers.append(model_arch.activation.out())
        else:
            width = int(concrete_layer_arch[i].hidden + concrete_layer_arch[i].extra)
            layers.append(nn.Linear(last_width,width,bias=model_arch.use_bias.hidden))
            layers.append(model_arch.activation.hidden())
            last_width = width

    policy_net = nn.Sequential(*layers)
    agent = TorchDDQN(
        sequential_list=policy_net,
        state_size=env.state_size,
        action_size=env.action_size,
        lr=hp.learning_rate,
        gamma=hp.discount_factor,
        batch_size=hp.batch_size,
        epsilon_start=1.0,
        epsilon_final=0.1,
        epsilon_decay=hp.epsilon_decay,
        learn_interval=hp.steps_learn_interval,
        min_replay_size=max(1000,2*hp.batch_size),
        use_per=getattr(hp, "use_per", False),
        per_alpha=getattr(hp, "per_alpha", 0.6),
        per_beta_start=getattr(hp, "per_beta_start", 0.4),
        per_beta_final=getattr(hp, "per_beta_final", 1.0),
        per_beta_steps=getattr(hp, "per_beta_steps", 100_000),
        per_eps=getattr(hp, "per_eps", 1e-6),
    )
    #agent.compile()
    parameters_cnt = sum(p.numel() for p in agent.policy_net.parameters())

    agent_metrics = ModelTrainMetrics()
    cum_goals     = 0
    goal_reached  = np.zeros((hp.episodes),dtype=bool)
    window_goal_count = 0
    consecutive_perfect = 0

    for episode in range(hp.episodes):
        epoch_start_time = time.perf_counter()
        cum_reward = 0.0
        cum_steps  = 0
        truncated  = False

        state_gpu = env.reset()               # (1, state_size) on GPU -- zero transfer
        state_np  = env.last_state_np         # (state_size,)   on CPU -- zero GPU sync

        for step in range(hp.max_steps):
            action                          = agent.act(state_gpu, eval=False)
            next_gpu, reward, done, extras  = env.step(action)

            if env.isGoal(extras["raw_ns"]):
                goal_reached[episode] = True
                cum_goals            += 1

            next_np = env.last_state_np       # CPU -- zero GPU sync
            # done=True only on goal (terminated). max_steps cap leaves the
            # last transition with done=False so bootstrap continues — the
            # canonical Gymnasium-style handling for truncation.
            agent.remember(state_np, action, reward, next_np, done)

            state_gpu = next_gpu              # GPU tensor -- zero transfer
            state_np  = next_np               # reuse; no recomputation next iter
            cum_reward += reward
            cum_steps   = step
            if done:
                break
        else:
            truncated = True

        agent.update_epsilon()
        loss_val = float(agent.loss)

        # Incremental rolling window success rate -- O(1) per episode
        window_goal_count += int(goal_reached[episode])
        success_rate = 0.0
        if episode >= hp.rolling_window_size:
            window_goal_count -= int(goal_reached[episode - hp.rolling_window_size])
            success_rate = (window_goal_count / hp.rolling_window_size) * 100

        delta_time = time.perf_counter() - epoch_start_time
        agent_metrics.append(episode,cum_reward,cum_goals,
                             success_rate,loss_val,cum_steps,
                             parameters_cnt,delta_time,model_arch.max_layers)

        # Early stopping: sustained 100% success rate
        if early_stop_episodes > 0:
            if success_rate >= 100.0:
                consecutive_perfect += 1
                if consecutive_perfect >= early_stop_episodes:
                    if verbose:
                        print(f"[INFO] Early stop at episode {episode} (100% for {early_stop_episodes} ep)")
                    break
            else:
                consecutive_perfect = 0

    agent.save(save_path)

    return agent_metrics


_TEMP_BASELINE_CACHE_DIR = "./TEMP_TEMP_BASELINE_CACHE_DIR/"

def train_thread(
        maze_path    : str,
        model_path   : str,
        metrics_path : str,
        train_fn     : str,
        train_tag    : str,
        hp           : GlobalHyperparameters,
        state_repr   : StateRepresentation,
        concrete_arch: List[LayersConfig],
        model_arch   : ModelArch,
        mode_type    : LayerModeType,
        mutation_mode: MutationMode,
        runs         : int,
        verbose      : bool = True,
        maze_wrapper = GPUMazeWrapper,
        early_stop_episodes: int = 0,
        insertion_type = None
):
    _FN_REGISTRY = {
        "train_reserved_saecollab_tolerance_model": train_reserved_saecollab_tolerance_model,
        "train_baseline_dense_model":               train_baseline_dense_model,
        "train_saecollab_tolerance_model":          train_saecollab_tolerance_model,
    }

    fn        = _FN_REGISTRY[train_fn]
    mode_type = LayerModeType.from_tag(mode_type)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Build environment -------------------------------------------------
    # GPUMazeWrapper  -> constructed directly; returns GPU tensors natively.
    # MazeGymWrapper  -> wrapped in CPUMazeWrapperAdapter so training
    #                   functions (written against GPUMazeWrapper) need no changes.
    maze_env = MazeEnv(
        maze_path,
        reward_shaping=getattr(hp, "use_reward_shaping", False),
        shaping_gamma=hp.discount_factor,
    )
    if maze_wrapper is GPUMazeWrapper:
        env = GPUMazeWrapper(maze_env, device=device, **state_repr.opts)
    else:
        # Any CPU-based wrapper (e.g. MazeGymWrapper)
        gym_env = maze_wrapper(maze_env, **state_repr.opts)
        env     = CPUMazeWrapperAdapter(gym_env, device=device)

    should_cache_current_model_and_metrics = False
    cached_baseline_path = None
    if train_fn == "train_baseline_dense_model" and insertion_type is not None:
        import shutil, filelock
        from os.path import basename
        maze_tag      = basename(maze_path).split(".")[0]
        cached_baseline_path: Path = Path(_TEMP_BASELINE_CACHE_DIR) / maze_tag / state_repr.tag / model_arch.tag / insertion_type.tag
        lock_path = str(cached_baseline_path) + ".lock"
        os.makedirs(cached_baseline_path.parent, exist_ok=True)
        lock = filelock.FileLock(lock_path, timeout=7200)

        with lock:
            cached_model   = cached_baseline_path / "model.pth"
            cached_metrics = cached_baseline_path / "metrics.csv"
            if cached_model.exists() and cached_metrics.exists():
                # Cache hit -- copy and skip training
                dest_dir = Path(metrics_path).parent
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(cached_model),   str(dest_dir / "model.pth"))
                shutil.copy2(str(cached_metrics), str(dest_dir / "metrics.csv"))
                if verbose:
                    print(f"[CACHE HIT] Copied baseline from {cached_baseline_path} -> {dest_dir}")

                del env
                import gc; gc.collect()
                torch.cuda.empty_cache()
                return None

            # Cache miss -- this worker trains and populates the cache
            should_cache_current_model_and_metrics = True

    metrics = fn(
        model_path, env, model_arch, concrete_arch,
        hp, mode_type, mutation_mode, runs, verbose,
        early_stop_episodes
    )

    if metrics:
        metrics.save(metrics_path)

    if verbose:
        print(f"[{train_tag}] Training Complete")

    if cached_baseline_path and should_cache_current_model_and_metrics:
        import shutil, filelock
        lock_path = str(cached_baseline_path) + ".lock"
        lock = filelock.FileLock(lock_path, timeout=7200)
        with lock:
            cached_baseline_path.mkdir(parents=True, exist_ok=True)
            if Path(model_path).exists():
                shutil.copy2(str(model_path), str(cached_baseline_path / "model.pth"))
            if Path(metrics_path).exists():
                shutil.copy2(str(metrics_path), str(cached_baseline_path / "metrics.csv"))

    del env
    del metrics
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return None  # metrics already saved to disk; no need to return the object
