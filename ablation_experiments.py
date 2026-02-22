#!python3 ablation_experiments.py

import time
import threading
import multiprocessing

from datetime import datetime, timedelta
from typing import *
from os.path import basename
from Ablation import *
from models.QModels import exp_decay_factor_to,TorchDDQN, \
                           SAECollabDDQN,ReservedSAECollabDDQN,NewLayerCfg

from env.MazeEnv import MazeEnv
from env.MazeWrapper import StateEncoder, MazeGymWrapper

from functools import reduce
import torch.nn as nn
import numpy as np
import time

def log_experiment_time(start_time, current_iter, total_experiment_iters, save_path, first_current_iter):
    """
    Enhanced experiment progress logger with live-updating display.
    Uses ANSI escape codes to overwrite previous output.
    
    Args:
        start_time: Unix timestamp when experiment started
        current_iter: Current iteration index (0-based)
        total_experiment_iters: Total number of iterations
        save_path: Current experiment save path
    """
    import sys
    
    now = time.time()
    elapsed = now - start_time

    done = current_iter + 1
    remaining = total_experiment_iters - done
    
    offseted_done = (done - first_current_iter)
    
    avg_time_per_iter = elapsed / offseted_done if (done - first_current_iter) > 0 else 0.0
    eta = avg_time_per_iter * remaining
    
    progress_pct = (done / total_experiment_iters * 100) if total_experiment_iters > 0 else 0.0
    
    iters_per_sec = offseted_done / elapsed if elapsed > 0 else 0.0
    
    def fmt_time(t):
        """Format seconds into HH:MM:SS"""
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    
    def fmt_time_verbose(t):
        """Format seconds into verbose string"""
        d = int(t // 86400)
        h = int((t % 86400) // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        
        parts = []
        if d > 0:
            parts.append(f"{d}d")
        if h > 0 or d > 0:
            parts.append(f"{h}h")
        if m > 0 or h > 0 or d > 0:
            parts.append(f"{m}m")
        parts.append(f"{s}s")
        
        return " ".join(parts)
    
    bar_length = 30
    filled_length = int(bar_length * done / total_experiment_iters)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    
    separator = "=" * 80
    
    # ANSI escape codes
    # \033[F moves cursor up one line
    # \033[K clears from cursor to end of line
    # Count total lines in output (11 lines)
    num_lines = 14
    
    if current_iter > 0:
        sys.stdout.write(f"\033[{num_lines}F")  # Move up num_lines
    
    output = []
    output.append(separator)
    output.append("📊 EXPERIMENT PROGRESS")
    output.append(separator)
    output.append(f"Progress: [{bar}] {progress_pct:.1f}%")
    output.append(f"Iteration: {done:,} / {total_experiment_iters:,}")
    output.append("")
    output.append("⏱️  TIMING STATISTICS")
    output.append(f"  Elapsed:        {fmt_time(elapsed)} ({fmt_time_verbose(elapsed)})")
    output.append(f"  ETA:            {fmt_time(eta)} ({fmt_time_verbose(eta)})")
    output.append(f"  Avg/Iter:       {avg_time_per_iter:.2f}s")
    output.append(f"  Speed:          {iters_per_sec:.3f} iters/sec")
    output.append("")
    output.append(f"📁 Current Path: {save_path}")
    output.append(separator)
    
    for line in output:
        sys.stdout.write("\033[K")
        print(line)
    
    sys.stdout.flush()

def gen_concrete_arch(
    base_width    : int,
    env           : MazeGymWrapper,
    architecute   : ModelArch,
    insertion_type: LayerInsertionType
) -> Optional[List[LayersConfig]]:
    
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
            hidden_size = np.ceil(crt_hidden + i * delta_hidden)
            extra_size  = np.ceil(crt_extra + i * delta_extra)
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
            hidden_size = np.ceil(drt_hidden - i * delta_hidden)
            extra_size  = np.ceil(drt_extra - i * delta_extra)
            concrete_arch.append(
                LayersConfig(hidden_size,
                             env.action_size,
                             extra_size)
            )
        return concrete_arch
    
    elif insertion_type is LayerInsertionType.ALT:
        concrete_arch = list()
        drt_hidden   = architecute.width_multipliers.hidden * base_width
        drt_extra    = architecute.width_multipliers.extra  * base_width
        delta_hidden = architecute.delta_width_multipliers.hidden * drt_hidden
        delta_extra  = architecute.delta_width_multipliers.extra * drt_extra 
        sigma = 0.0
        for i in range(architecute.max_layers):
            if i > 0 : sigma = (2.0*np.random.random() - 1.0)
            drt_hidden += np.ceil(sigma * delta_hidden)
            drt_extra  += np.ceil(sigma * delta_extra)
            concrete_arch.append(
                LayersConfig(drt_hidden,
                             env.action_size,
                             drt_extra)
            )
        return concrete_arch
    
    else:
        print(f"[ERROR] Insertion Type Invalid: {insertion_type}")
        return None

def save_concrete_arch_info(save_dir:str,concrete_arch:List[LayersConfig]):
    concrete_arch_repr = {"arch": [
        {
            "hidden":layer.hidden,
            "out":layer.out,
            "extra":layer.extra
        } for layer in concrete_arch
    ]}
    with open(os.path.join(save_dir,"concrete_arch.json"),"w") as f:
        json.dump(concrete_arch_repr,f,indent=4)

def eval_agent_deterministic(agent, env, max_steps):
    """Avalia se agente consegue alcançar goal deterministicamente"""
    old_epsilon = getattr(agent, "epsilon", None)
    prev_training = getattr(agent.policy_net, "training", True)
    agent.policy_net.eval
    
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

def train_saecollab_tolerance_model(
    save_path:str,
    env: MazeGymWrapper,
    model_arch:ModelArch,
    concrete_layer_arch: List[LayersConfig],
    hp:GlobalHyperparameters,
    mode_type:LayerModeType,
    mutation_mode:MutationMode,
    runs:int
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
        use_bias=model_arch.use_bias
    )

    parameters_cnt = sum(p.numel() for p in agent.policy_net.parameters())
    agent_metrics  = ModelTrainMetrics()
    cum_goals      = 0
    goal_reached   = np.zeros((hp.episodes),dtype=bool)
    current_branch = 0
    deterministic_reached = False
    episodes_since_last_branch = 0
    # Check if the goal at least once was reached
    goal_once_reached = False
    for episode in range(hp.episodes):
        epoch_start_time = datetime.now()
        cum_reward = 0.0
        cum_steps  = 0
        state = env.reset()
        state = state.reshape(1,-1)
        
        for step in range(hp.max_steps):
            action = agent.act(state, eval=False)
            next_state, reward, done, extras_dict = env.step(action)
            if env.isGoal(extras_dict["raw_ns"]):
                goal_reached[episode] = True
                goal_once_reached = True
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
                loss_val = float(agent.loss.cpu().detach())
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
        episodes_since_last_branch += 1
        if (episode >= hp.insert_patience and
           episode % hp.insert_patience == 0 and
           current_branch < model_arch.max_layers):
           
           if deterministic_reached or (_dr:=eval_agent_deterministic(agent,env,hp.max_steps)):
                if not deterministic_reached and _dr: 
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
                   extra_size  = int(concrete_layer_arch[current_branch].extra if mode_type.value.use_extra_branch else 0)
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
        epoch_end_time = datetime.now()
        delta_time = (epoch_end_time - epoch_start_time).total_seconds()
        agent_metrics.append(episode,cum_reward,cum_goals,success_rate,
                             loss_val,cum_steps,parameters_cnt,
                             delta_time,current_branch
                             )        

        #if episode % 50 == 0:
        #    agent_metrics.pretty_print(5)


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
    runs:int
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
        use_bias              = model_arch.use_bias
    )

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

    for episode in range(hp.episodes):
        epoch_start_time = datetime.now()
        cum_reward = 0.0
        cum_steps  = 0
        state = env.reset()
        state = state.reshape(1, -1)

        for step in range(hp.max_steps):
            action = agent.act(state, eval=False)
            next_state, reward, done, extras_dict = env.step(action)
            if env.isGoal(extras_dict["raw_ns"]):
                goal_reached[episode] = True
                goal_once_reached     = True
                cum_goals            += 1
            next_state = next_state.reshape(1, -1)
            agent.remember(state, action, reward, next_state, done)
            state      = next_state
            cum_reward += reward
            cum_steps   = step
            if done:
                break

        agent.policy_net.step_all_etas()
        agent.update_epsilon()

        if hasattr(agent, 'loss'):
            if isinstance(agent.loss, torch.Tensor):
                loss_val = float(agent.loss.cpu().detach())
            else:
                loss_val = float(agent.loss) if agent.loss is not None else 0.0
        else:
            loss_val = 0.0

        # Success Rate Calculation
        success_rate = 0.0
        if episode >= hp.rolling_window_size:
            start_idx     = max(0, episode - hp.rolling_window_size + 1)
            recent_window = goal_reached[start_idx: episode + 1]
            recent_goals  = int(np.sum(recent_window))
            success_rate  = (recent_goals / hp.rolling_window_size) * 100

        # New Branch Logic — same tolerance criteria, but uses use_next_layer()
        episodes_since_last_branch += 1
        max_branches = model_arch.max_layers - 1  # first layer already active

        if (episode >= hp.insert_patience and
            episode % hp.insert_patience == 0 and
            current_branch < max_branches):

            if deterministic_reached or (_dr := eval_agent_deterministic(agent, env, hp.max_steps)):
                if not deterministic_reached and _dr:
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

        epoch_end_time = datetime.now()
        delta_time     = (epoch_end_time - epoch_start_time).total_seconds()
        agent_metrics.append(episode, cum_reward, cum_goals, success_rate,
                             loss_val, cum_steps, parameters_cnt,
                             delta_time, current_branch)

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
                loss_val = float(agent.loss.cpu().detach())
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
    runs:int
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
    )
    parameters_cnt = sum(p.numel() for p in agent.policy_net.parameters())
    
    agent_metrics = ModelTrainMetrics()
    cum_goals     = 0
    goal_reached  = np.zeros((hp.episodes),dtype=bool) 
    
    for episode in range(hp.episodes):
        epoch_start_time = datetime.now()
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
        
        agent.update_epsilon()

        # Convert loss to Python float, handling CUDA tensors
        if hasattr(agent, 'loss'):
            if isinstance(agent.loss, torch.Tensor):
                loss_val = float(agent.loss.cpu().detach())
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

        epoch_end_time = datetime.now()
        delta_time = (epoch_end_time - epoch_start_time).total_seconds()
        agent_metrics.append(episode,cum_reward,cum_goals,
                             success_rate,loss_val,cum_steps,
                             parameters_cnt,delta_time,model_arch.max_layers)

        #if episode % 50 == 0:
        #    agent_metrics.pretty_print(5)

    agent.save(save_path)

    return agent_metrics

def train_thread(
        maze_path:str,
        model_path:str,
        metrics_path:str,
        train_fn:Callable,
        train_tag:str,
        hp: GlobalHyperparameters,
        state_repr: StateRepresentation,
        concrete_arch: List[LayersConfig],
        model_arch: ModelArch,
        mode_type: LayerModeType,
        mutation_mode: MutationMode,
        runs: int
):
        # Create separate environment for this thread
        env = MazeGymWrapper(MazeEnv(maze_path),**state_repr.opts)
        mode_type = LayerModeType.from_tag(mode_type)

        metrics = train_fn(
            model_path,
            env,
            model_arch,
            concrete_arch,
            hp,
            mode_type,
            mutation_mode,
            runs
        )
        
        if metrics:
            metrics.save(metrics_path)
        
        print(f"[{train_tag}] Training Complete")

        return metrics

def train_models(
    state: AblationProgramState,
    maze: MazeEnv,
    hp: GlobalHyperparameters,
    state_repr: StateRepresentation,
    architecute: ModelArch,
    insertion_type: LayerInsertionType,
    mode_type: LayerModeType,
    mutation_mode: MutationMode,
    runs: int,
    processig_mode:Literal['sequential','threading','multiprocessing'] = 'multiprocessing'
):
    gym_env_maze = MazeGymWrapper(
        maze,
        **state_repr.opts
    )
    tabular_like_width = gym_env_maze.action_size * gym_env_maze.rows \
                         * gym_env_maze.cols 
    concrete_arch = gen_concrete_arch(tabular_like_width,
                                      gym_env_maze,
                                      architecute,
                                      insertion_type)
    save_concrete_arch_info(state.save_dir_path,concrete_arch)
    
    def create_out_paths(model_name:str) -> None:
        model_dir = os.path.join(state.save_dir_path,model_name)
        os.makedirs(model_dir,exist_ok=True)    
        model_path   = os.path.join(model_dir,"model.pth")
        metrics_path = os.path.join(model_dir,"metrics.csv")
        return model_path,metrics_path

    # Prepare directories and paths
    dense_model_path,dense_metrics_path = create_out_paths("dense_model")
    sae_tolerance_model_path,sae_tolerance_metrics_path = create_out_paths("sae_tolerance_model")
    sae_spaced_model_path,sae_spaced_metrics_path = create_out_paths("sae_spaced_model")

    train_targets = [
        TrainTargetClosure(
            tag="SAE Tolerance",
            fn=train_reserved_saecollab_tolerance_model,
            save_model_path=sae_tolerance_model_path,
            save_metrics_path=sae_tolerance_metrics_path,
        ),
        TrainTargetClosure(
            tag="Baseline",
            fn=train_baseline_dense_model,
            save_model_path=dense_model_path,
            save_metrics_path=dense_metrics_path
        ),
    ]

    # Comparation of SAE Tolerance methods
    # reserved_tolerance_model_path,reserved_tolerance_metrics_path = create_out_paths("reserved_sae_tolerance_model")
    # train_targets = [
    #     TrainTargetClosure(
    #         tag="Reserved SAE Tolerance",
    #         fn=train_reserved_saecollab_tolerance_model,
    #         save_model_path=reserved_tolerance_model_path,
    #         save_metrics_path=reserved_tolerance_metrics_path
    #     ),
    #     TrainTargetClosure(
    #         tag="Dyn SAE Tolerance",
    #         fn=train_saecollab_tolerance_model,
    #         save_model_path=sae_tolerance_model_path,
    #         save_metrics_path=sae_tolerance_metrics_path,
    #     )
    # ]    

    args = {
        "maze_path"    : maze.file_path,
        "hp"           : hp,
        "state_repr"   : state_repr,
        "concrete_arch": concrete_arch,
        "model_arch"   : architecute,
        "mode_type"    : mode_type.tag,
        "mutation_mode": mutation_mode,
        "runs"         : runs
    }

    if processig_mode == "sequential":
        for target in train_targets:
            args["model_path"]   = target.save_model_path
            args["metrics_path"] = target.save_metrics_path
            args["train_fn"]     = target.fn
            args["train_tag"]    = target.tag

            train_thread(**args)
            
        print("[INFO] Sequential training completed")

    if processig_mode == 'multiprocessing':
        process_poll = []
        for target in train_targets:
            args["model_path"]   = target.save_model_path
            args["metrics_path"] = target.save_metrics_path
            args["train_fn"]     = target.fn
            args["train_tag"]    = target.tag

            p = multiprocessing.Process(target=train_thread,kwargs=args)
            process_poll.append(p)
            p.start()
        
        for p in process_poll:
            p.join()

        print("[INFO] Multiprocessing training completed")
        

    if processig_mode == 'threading':
        thread_poll = []

        for target in train_targets:
            args["model_path"]   = target.save_model_path
            args["metrics_path"] = target.save_metrics_path
            args["train_fn"]     = target.fn
            args["train_tag"]    = target.tag

            p = threading.Thread(target=train_thread,name=target.tag,kwargs=args)
            thread_poll.append(p)
            p.start()
        
        for p in thread_poll:
            p.join()

        print("[INFO] Threading training completed")

def experiment_1(dir_path:str=None,
                 seed=None,
                 TABULAR_QLEARNING_PATH = "./c_qlearning/build/agentTrain.exe",
                 processig_mode="sequential"
                 ):
    dir_path = dir_path or 'experiment_1'
    seed     = seed or 333
    state = AblationProgramState.load_from_json(dir_path,seed)
    set_seed(seed)
    start_time = time.time()

    if state is None:
        state = AblationProgramState(
            TABULAR_QLEARNING_PATH,
            dir_path,
            seed
        )
        state.env_update()

    # COMBINATORIAL OPTIONS START
    
    MAZES = [
        "./mazes/small_eg.maze",
        "./mazes/medium_eg.maze",
        "./mazes/big_eg.maze"
    ]

    # USED BY ENV WRAPPERS 
    STATE_REPRESENTATIONS = [
        StateRepresentation(
            state_encoder=StateEncoder.ONE_HOT
        ),
        StateRepresentation(
            state_encoder=StateEncoder.COORDS,
            possible_actions_feature= True
        ),
        StateRepresentation(
            state_encoder=StateEncoder.COORDS_NORM,
            num_last_states=2,
            visited_count=True
        ),
        StateRepresentation(
            state_encoder=StateEncoder.ONE_HOT,
            possible_actions_feature=True,
        ),
    ]

    N_MAX_LAYERS = 4 
    # GARANTINDO QUE ATE O ULTIMO LAYER TENHAM VARIAÇÕES VALIDAS
    width_delta  = 1 / (N_MAX_LAYERS)
    ARCHITECTURES = [
        ModelArch(N_MAX_LAYERS,
                  LayersConfig(1/2,1,1/2),
                  LayersConfig(width_delta,1,width_delta),
                  LayersConfig(nn.ReLU,nn.Identity,nn.ReLU),
                  LayersConfig(True,True,True)
                  ),
        ModelArch(N_MAX_LAYERS,
                  LayersConfig(1/2,1,1/4),
                  LayersConfig(width_delta,1,width_delta),
                  LayersConfig(nn.ReLU,nn.Identity,nn.Identity),
                  LayersConfig(True,True,True)
                  ),

        ModelArch(N_MAX_LAYERS,
                  LayersConfig(1/4,1,1/4),
                  LayersConfig(width_delta,1,width_delta),
                  LayersConfig(nn.ReLU,nn.Identity,nn.Identity),
                  LayersConfig(False,True,False)
                  ),

        ModelArch(N_MAX_LAYERS,
                  LayersConfig(1/4,1,1/2),
                  LayersConfig(width_delta,1,width_delta),
                  LayersConfig(nn.ReLU,nn.Identity,nn.ReLU),
                  LayersConfig(False,True,False)
                  ),
    ]

    INSERTION_TYPES = list(LayerInsertionType)
    LAYER_MODES     = list(LayerModeType)[::-1]
    MUTATION_MODES  = list(MutationMode)

    # VALIDA PARA O MODO TABULAR TAMBEM
    runs = 1

    COMB_ARRAYS_LIST = [MAZES, STATE_REPRESENTATIONS, ARCHITECTURES, INSERTION_TYPES, LAYER_MODES, MUTATION_MODES]
    DIMENSIONS = [len(arr) for arr in COMB_ARRAYS_LIST]
    TOTAL_EXPERIMENT_ITERS = reduce(lambda x,y : x * y, DIMENSIONS)
    
    # Get starting indices for resuming
    skip_indices = state.get_skip_indices(DIMENSIONS)
    print(f"[INFO] Starting from indices: {skip_indices}")
    print(f"[INFO] Completed iterations: {state.completed_iteration + 1}/{TOTAL_EXPERIMENT_ITERS}")

    # COMBINATORIAL OPTIONS END

    for maze_idx, maze_path in enumerate(MAZES):
        if maze_idx < skip_indices[0]:
            continue
            
        maze_tag = basename(maze_path).split(".")[0]
        state.add_save_path_head(maze_tag)   
        maze_env = MazeEnv(maze_path, rewards_scaled=False, pass_through_walls=False)
        
        # GLOBAL HYPERPARAMETERS
        EPISODES  = 400
        MAX_STEPS = maze_env.rows * maze_env.cols * len(list(Action))
        # MAX_STEPS = maze_env.opens_count * len(list(Action))
        
        epsilon_decay = exp_decay_factor_to(
                final_epsilon=0.1,
                final_step=MAX_STEPS * EPISODES,
                epsilon_start=1.0,
                convergence_threshold=0.01
        )
        
        hyperparameters = GlobalHyperparameters(
            learning_rate           = 1e-5,
            new_layer_learning_rate = 5e-5,
            discount_factor         = 0.999,
            epsilon_decay           = epsilon_decay,
            episodes                = EPISODES,
            max_steps               = MAX_STEPS,
            batch_size              = 512,
            steps_learn_interval    = 4,
            rolling_window_size     = 20,
            insert_patience         = 15,
            insert_min_goals        = 5,
            insert_min_variance     = 0.6
        )

        # TRAIN TABULAR MODEL
        try:
            state.train_tabular_agent(maze_path,hyperparameters,runs)
        except Exception as e:
            print(f"[WARNING] Cant Train Tabular Agent: {e}")
        state.save_a_star_qtable(maze_path)
        
        first_current_iter = state.completed_iteration

        for state_rep_idx, state_representation in enumerate(STATE_REPRESENTATIONS):
            if maze_idx == skip_indices[0] and state_rep_idx < skip_indices[1]:
                continue
                
            repr_tag = state_representation.tag
            state.add_save_path_head(repr_tag)   
            
            for arch_idx, arch in enumerate(ARCHITECTURES):
                if maze_idx == skip_indices[0] and state_rep_idx == skip_indices[1] and arch_idx < skip_indices[2]:
                    continue
                    
                arch_tag = arch.tag
                state.add_save_path_head(arch_tag)   
                
                for insert_idx, insertion_type in enumerate(INSERTION_TYPES):
                    if (maze_idx == skip_indices[0] and state_rep_idx == skip_indices[1] and 
                        arch_idx == skip_indices[2] and insert_idx < skip_indices[3]):
                        continue
                        
                    insert_tag = insertion_type.tag
                    state.add_save_path_head(insert_tag)   

                    for layer_mode_idx, layer_mode in enumerate(LAYER_MODES):
                        if (maze_idx == skip_indices[0] and state_rep_idx == skip_indices[1] and 
                            arch_idx == skip_indices[2] and insert_idx == skip_indices[3] and
                            layer_mode_idx < skip_indices[4]):
                            continue
                            
                        layer_tag = layer_mode.tag
                        state.add_save_path_head(layer_tag)   
                        
                        for mutation_idx, mutation_mode in enumerate(MUTATION_MODES):
                            if (maze_idx == skip_indices[0] and state_rep_idx == skip_indices[1] and 
                                arch_idx == skip_indices[2] and insert_idx == skip_indices[3] and
                                layer_mode_idx == skip_indices[4] and mutation_idx < skip_indices[5]):
                                continue
                                
                            mutation_tag = str(mutation_mode).split(".")[-1]
                            state.add_save_path_head(mutation_tag)
                            
                            current_indices = [maze_idx, state_rep_idx, arch_idx, insert_idx, layer_mode_idx, mutation_idx]
                            current_iter = state.get_current_iteration(current_indices, DIMENSIONS)
                            
                            log_experiment_time(start_time,current_iter,TOTAL_EXPERIMENT_ITERS,state.save_dir_path,first_current_iter)

                            # MODELS TRAINING START HERE

                            train_models(state,
                                         maze_env,
                                         hyperparameters,
                                         state_representation,
                                         arch,
                                         insertion_type,
                                         layer_mode,
                                         mutation_mode,
                                         runs,
                                         processig_mode=processig_mode
                                         )

                            # MODELS TRAINING ENDS HERE
                            
                            state.mark_iteration_complete(current_iter)

                            state.remove_save_path_head()
                        
                        state.remove_save_path_head()

                    state.remove_save_path_head()

                state.remove_save_path_head()

            state.remove_save_path_head()

        state.remove_save_path_head()

def fast_experiment_1(dir_path:str=None,
                      seed=None,
                      TABULAR_QLEARNING_PATH = "./c_qlearning/build/agentTrain.exe",
                      max_workers=16,
                      **kwargs):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    dir_path = dir_path or 'experiment_1'
    state = AblationProgramState.load_from_json(dir_path,seed)
    seed     = seed or 333
    set_seed(seed)

    start_time = time.time()

    if state is None:
        state = AblationProgramState(
            TABULAR_QLEARNING_PATH,
            dir_path,
            seed
        )
        state.env_update()

    # COMBINATORIAL OPTIONS START
    
    MAZES = [
        "./mazes/small_eg.maze",
        "./mazes/medium_eg.maze",
        "./mazes/big_eg.maze"
    ]

    # USED BY ENV WRAPPERS 
    STATE_REPRESENTATIONS = [
        StateRepresentation(
            state_encoder=StateEncoder.ONE_HOT
        ),
        StateRepresentation(
            state_encoder=StateEncoder.COORDS,
            possible_actions_feature= True
        ),
        StateRepresentation(
            state_encoder=StateEncoder.COORDS_NORM,
            num_last_states=2,
            visited_count=True
        ),
        StateRepresentation(
            state_encoder=StateEncoder.ONE_HOT,
            possible_actions_feature=True,
        ),
    ]

    N_MAX_LAYERS = 4 
    # GARANTINDO QUE ATE O ULTIMO LAYER TENHAM VARIAÇÕES VALIDAS
    width_delta  = 1 / (N_MAX_LAYERS)
    ARCHITECTURES = [
        ModelArch(N_MAX_LAYERS,
                  LayersConfig(1/2,1,1/2),
                  LayersConfig(width_delta,1,width_delta),
                  LayersConfig(nn.ReLU,nn.Identity,nn.ReLU),
                  LayersConfig(True,True,True)
                  ),
        ModelArch(N_MAX_LAYERS,
                  LayersConfig(1/2,1,1/4),
                  LayersConfig(width_delta,1,width_delta),
                  LayersConfig(nn.ReLU,nn.Identity,nn.Identity),
                  LayersConfig(True,True,True)
                  ),

        ModelArch(N_MAX_LAYERS,
                  LayersConfig(1/4,1,1/4),
                  LayersConfig(width_delta,1,width_delta),
                  LayersConfig(nn.ReLU,nn.Identity,nn.Identity),
                  LayersConfig(False,True,False)
                  ),

        ModelArch(N_MAX_LAYERS,
                  LayersConfig(1/4,1,1/2),
                  LayersConfig(width_delta,1,width_delta),
                  LayersConfig(nn.ReLU,nn.Identity,nn.ReLU),
                  LayersConfig(False,True,False)
                  ),
    ]

    INSERTION_TYPES = list(LayerInsertionType)
    LAYER_MODES     = list(LayerModeType)[::-1]
    MUTATION_MODES  = list(MutationMode)

    # VALIDA PARA O MODO TABULAR TAMBEM
    runs = 1

    COMB_ARRAYS_LIST = [MAZES, STATE_REPRESENTATIONS, ARCHITECTURES, INSERTION_TYPES, LAYER_MODES, MUTATION_MODES]
    DIMENSIONS = [len(arr) for arr in COMB_ARRAYS_LIST]
    TOTAL_EXPERIMENT_ITERS = reduce(lambda x,y : x * y, DIMENSIONS)
    
    # Get starting indices for resuming
    skip_indices = state.get_skip_indices(DIMENSIONS)
    print(f"[INFO] Starting from indices: {skip_indices}")
    print(f"[INFO] Completed iterations: {state.completed_iteration + 1}/{TOTAL_EXPERIMENT_ITERS}")

    # COMBINATORIAL OPTIONS END    

    all_jobs = []

    for maze_idx, maze_path in enumerate(MAZES):
        maze_env = MazeEnv(maze_path, rewards_scaled=False, pass_through_walls=False)

        # GLOBAL HYPERPARAMETERS
        EPISODES  = 400
        MAX_STEPS = maze_env.rows * maze_env.cols * len(list(Action))
        # MAX_STEPS = maze_env.opens_count * len(list(Action))
        
        epsilon_decay = exp_decay_factor_to(
                final_epsilon=0.1,
                final_step=MAX_STEPS * EPISODES,
                epsilon_start=1.0,
                convergence_threshold=0.01
        )
        
        hyperparameters = GlobalHyperparameters(
            learning_rate           = 1e-5,
            new_layer_learning_rate = 5e-5,
            discount_factor         = 0.999,
            epsilon_decay           = epsilon_decay,
            episodes                = EPISODES,
            max_steps               = MAX_STEPS,
            batch_size              = 512,
            steps_learn_interval    = 4,
            rolling_window_size     = 20,
            insert_patience         = 15,
            insert_min_goals        = 5,
            insert_min_variance     = 0.6
        )

        # TRAIN TABULAR MODEL
        try:
            state.train_tabular_agent(maze_path,hyperparameters,runs)
        except Exception as e:
            print(f"[WARNING] Cant Train Tabular Agent: {e}")
        state.save_a_star_qtable(maze_path)
        
        for state_representation in STATE_REPRESENTATIONS:
            for arch in ARCHITECTURES:
                for insertion_type in INSERTION_TYPES:
                    for layer_mode in LAYER_MODES:
                        for mutation_mode in MUTATION_MODES:

                            gym_env = MazeGymWrapper(maze_env, **state_representation.opts)
                            base_width = gym_env.action_size * maze_env.rows * maze_env.cols
                            concrete_arch = gen_concrete_arch(base_width, gym_env, arch, insertion_type)

                            save_dir = os.path.join(
                                dir_path, str(seed),
                                basename(maze_path).split(".")[0],
                                state_representation.tag,
                                arch.tag,
                                insertion_type.tag,
                                layer_mode.tag,
                                str(mutation_mode).split(".")[-1]
                            )
                            os.makedirs(save_dir, exist_ok=True)

                            for train_fn, subdir in [
                                (train_reserved_saecollab_tolerance_model, "sae_tolerance_model"),
                                (train_baseline_dense_model,               "dense_model"),
                            ]:
                                model_dir    = os.path.join(save_dir, subdir)
                                model_path   = os.path.join(model_dir, "model.pth")
                                metrics_path = os.path.join(model_dir, "metrics.csv")
                                os.makedirs(model_dir, exist_ok=True)

                                if not os.path.exists(model_path):
                                    all_jobs.append({
                                        "maze_path"    : maze_path,
                                        "model_path"   : model_path,
                                        "metrics_path" : metrics_path,
                                        "train_fn"     : train_fn,
                                        "train_tag"    : subdir,
                                        "hp"           : hyperparameters,
                                        "state_repr"   : state_representation,
                                        "concrete_arch": concrete_arch,
                                        "model_arch"   : arch,
                                        "mode_type"    : layer_mode.tag,
                                        "mutation_mode": mutation_mode,
                                        "runs"         : 1
                                    })

    total = len(all_jobs)
    done  = 0
    print(f"[INFO] Total jobs: {total}")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(train_thread, **job): job for job in all_jobs}
        for future in as_completed(futures):
            done += 1
            print(f"[{done}/{total}] done", end="\r")
            try:
                future.result()
            except Exception as e:
                print(f"\n[ERROR] {futures[future]['model_path']}: {e}")

    print(f"\n[INFO] All {total} jobs complete")



SELECTABLE_EXPERIMENTS = [experiment_1,fast_experiment_1] 