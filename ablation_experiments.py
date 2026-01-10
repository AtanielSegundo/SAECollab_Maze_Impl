#!python3 ablation_experiments.py

import time

from os.path import basename
from Ablation import *
from models.QModels import exp_decay_factor_to

from env.MazeEnv import *
from env.MazeWrapper import StateEncoder, MazeGymWrapper

from functools import reduce
import torch.nn as nn


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

def experiment_1(dir_path:str=None,seed=None):
    dir_path = dir_path or 'experiment_1'
    seed     = seed or 333
    TABULAR_QLEARNING_PATH = "./c_qlearning/build/agentTrain.exe"
    
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
    
    MAZES = ["./mazes/small_eg.maze","./mazes/medium_eg.maze","./mazes/big_eg.maze"]

    # USED BY ENV WRAPPERS 
    STATE_REPRESENTATIONS = [
        StateRepresentation(
            state_encoder=StateEncoder.COORDS,
            possible_actions_feature= True
        ),
        StateRepresentation(
            state_encoder=StateEncoder.ONE_HOT
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

    ARCHITECTURES = [
        ModelArch(4,
                  LayersConfig(1/4,1,1/4),
                  LayersConfig(1/2,1,1/2),
                  LayersConfig(nn.ReLU(),nn.Identity(),nn.ReLU()),
                  LayersConfig(True,True,True)
                  ),
        ModelArch(4,
                  LayersConfig(1/4,1,1/8),
                  LayersConfig(1/2,1,1/2),
                  LayersConfig(nn.ReLU(),nn.Identity(),nn.Identity()),
                  LayersConfig(True,True,True)
                  ),

        ModelArch(4,
                  LayersConfig(1/4,1,1/8),
                  LayersConfig(1/2,1,1/2),
                  LayersConfig(nn.ReLU(),nn.Identity(),nn.Identity()),
                  LayersConfig(False,True,False)
                  ),

        ModelArch(4,
                  LayersConfig(1/4,1,1/4),
                  LayersConfig(1/2,1,1/2),
                  LayersConfig(nn.ReLU(),nn.Identity(),nn.ReLU()),
                  LayersConfig(False,True,False)
                  ),
    ]

    INSERTION_TYPES = list(LayerInsertionType)
    LAYER_MODES     = list(LayerModeType)
    MUTATION_MODES  = list(MutationMode)

    
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
        EPISODES  = 200
        MAX_STEPS = maze_env.opens_count * len(list(Action))
        
        epsilon_decay = exp_decay_factor_to(
                final_epsilon=0.1,
                final_step=MAX_STEPS,
                epsilon_start=1.0,
                convergence_threshold=0.01
            )
        
        hyperparameters = GlobalHyperparameters(
            learning_rate   = 1e-5,
            discount_factor = 0.999,
            epsilon_decay   = epsilon_decay,
            episodes        = EPISODES,
            max_steps       = MAX_STEPS
        )

        # TRAIN TABULAR MODEL
        state.train_tabular_agent(maze_path,hyperparameters)
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


                            # MODELS TRAINING ENDS HERE
                            
                            state.mark_iteration_complete(current_iter)

                            state.remove_save_path_head()
                        
                        state.remove_save_path_head()

                    state.remove_save_path_head()

                state.remove_save_path_head()

            state.remove_save_path_head()

        state.remove_save_path_head()

        

SELECTABLE_EXPERIMENTS = [experiment_1]