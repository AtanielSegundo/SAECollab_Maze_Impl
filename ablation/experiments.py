#!python3 ablation/experiments.py

import os
import sys
import time
import threading
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

from typing import *
from os.path import basename
from functools import reduce

import torch
import torch.nn as nn
import numpy as np

from ablation.config import (
    GlobalHyperparameters, ModelArch, LayerInsertionType,
    LayerModeType, StateRepresentation,
)
from ablation.metrics import ModelTrainMetrics, TrainTargetClosure
from ablation.state import AblationProgramState, set_seed
from ablation.training import (
    gen_concrete_arch, save_concrete_arch_info,
    train_saecollab_tolerance_model,
    train_reserved_saecollab_tolerance_model,
    train_saecollab_spaced_model,
    train_baseline_dense_model,
    train_thread,
    _TEMP_BASELINE_CACHE_DIR,
)

from models.QModels import exp_decay_factor_to
from env.MazeEnv import MazeEnv, Action
from env.MazeWrapper import StateEncoder, MazeGymWrapper
from env.GPUMazeWrapper import GPUMazeWrapper
from StackedCollab.collabNet import LayersConfig, MutationMode


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
    bar = '\u2588' * filled_length + '\u2591' * (bar_length - filled_length)

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
    output.append("\U0001f4ca EXPERIMENT PROGRESS")
    output.append(separator)
    output.append(f"Progress: [{bar}] {progress_pct:.1f}%")
    output.append(f"Iteration: {done:,} / {total_experiment_iters:,}")
    output.append("")
    output.append("\u23f1\ufe0f  TIMING STATISTICS")
    output.append(f"  Elapsed:        {fmt_time(elapsed)} ({fmt_time_verbose(elapsed)})")
    output.append(f"  ETA:            {fmt_time(eta)} ({fmt_time_verbose(eta)})")
    output.append(f"  Avg/Iter:       {avg_time_per_iter:.2f}s")
    output.append(f"  Speed:          {iters_per_sec:.3f} iters/sec")
    output.append("")
    output.append(f"\U0001f4c1 Current Path: {save_path}")
    output.append(separator)

    for line in output:
        sys.stdout.write("\033[K")
        print(line)

    sys.stdout.flush()


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
    processig_mode:Literal['sequential','threading','multiprocessing'] = 'multiprocessing',
    maze_wrapper = GPUMazeWrapper,
):
    gym_env_maze = MazeGymWrapper(
        maze,
        **state_repr.opts
    )

    tabular_like_width = gym_env_maze.action_size * gym_env_maze.rows * gym_env_maze.cols

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
    dense_model_path        ,dense_metrics_path         = create_out_paths("dense_model")
    sae_tolerance_model_path,sae_tolerance_metrics_path = create_out_paths("sae_tolerance_model")
    sae_spaced_model_path   ,sae_spaced_metrics_path    = create_out_paths("sae_spaced_model")

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

    args = {
        "maze_path"    : maze.file_path,
        "hp"           : hp,
        "state_repr"   : state_repr,
        "concrete_arch": concrete_arch,
        "model_arch"   : architecute,
        "mode_type"    : mode_type.tag,
        "mutation_mode": mutation_mode,
        "runs"         : runs,
        "maze_wrapper" : maze_wrapper,
    }

    if processig_mode == "sequential":
        for target in train_targets:
            args["model_path"]   = target.save_model_path
            args["metrics_path"] = target.save_metrics_path
            args["train_fn"]     = target.fn.__name__
            args["train_tag"]    = target.tag

            train_thread(**args)

        print("[INFO] Sequential training completed")

    if processig_mode == 'multiprocessing':
        process_poll = []
        for target in train_targets:
            args["model_path"]   = target.save_model_path
            args["metrics_path"] = target.save_metrics_path
            args["train_fn"]     = target.fn.__name__
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
            args["train_fn"]     = target.fn.__name__
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
                 processig_mode="sequential",
                 maze_wrapper=GPUMazeWrapper,
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
        "./mazes/big_eg.maze",
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
    # GARANTINDO QUE ATE O ULTIMO LAYER TENHAM VARIACOES VALIDAS
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
        #MAX_STEPS = maze_env.rows * maze_env.cols * len(list(Action))

        MAX_STEPS = len(list(Action)) * maze_env.opens_count
        MAX_STEPS = int(MAX_STEPS)

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
                                         processig_mode=processig_mode,
                                         maze_wrapper=maze_wrapper,
                                         )

                            # MODELS TRAINING ENDS HERE

                            state.mark_iteration_complete(current_iter)

                            state.remove_save_path_head()

                        state.remove_save_path_head()

                    state.remove_save_path_head()

                state.remove_save_path_head()

            state.remove_save_path_head()

        state.remove_save_path_head()

# -- Worker warm-up --------------------------------------------------------
# Must be a top-level function so multiprocessing can pickle it.
def _warm_worker():
    import torch
    # Force CUDA context creation right now.
    if torch.cuda.is_available():
        _dummy = torch.zeros(1, device="cuda")
        del _dummy
        torch.cuda.empty_cache()
    # Pre-import the heavy project modules
    try:
        from models.QModels import TorchDDQN, SAECollabDDQN, ReservedSAECollabDDQN   # noqa
        from env.MazeEnv import MazeEnv                                              # noqa
        from env.GPUMazeWrapper import GPUMazeWrapper                                # noqa
        from env.CPUMazeWrapper import CPUMazeWrapperAdapter                         # noqa
    except ImportError:
        pass


def fast_experiment_1(
    dir_path: str = None,
    seed=None,
    TABULAR_QLEARNING_PATH="./c_qlearning/build/agentTrain.exe",
    max_workers=6,
    learns_by_epochs=None,
    maze_wrapper=GPUMazeWrapper,
    **kwargs,
):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from collections import deque
    import traceback

    dir_path = dir_path or "experiment_1"
    seed = seed or 333
    set_seed(seed)

    os.makedirs(_TEMP_BASELINE_CACHE_DIR,exist_ok=True)

    state = AblationProgramState.load_from_json(dir_path, seed)
    if state is None:
        state = AblationProgramState(TABULAR_QLEARNING_PATH, dir_path, seed)
        state.env_update()

    # -- Experiment configuration ------------------------------------------
    MAZES = [
        "./mazes/big_eg.maze",
        "./mazes/small_eg.maze",
        "./mazes/medium_eg.maze",
    ]

    STATE_REPRESENTATIONS = [
        StateRepresentation(state_encoder=StateEncoder.ONE_HOT),
        StateRepresentation(state_encoder=StateEncoder.COORDS, possible_actions_feature=True),
        StateRepresentation(state_encoder=StateEncoder.COORDS_NORM, num_last_states=2, visited_count=True),
        StateRepresentation(state_encoder=StateEncoder.ONE_HOT, possible_actions_feature=True),
    ]

    N_MAX_LAYERS = 4
    width_delta = 1 / N_MAX_LAYERS
    ARCHITECTURES = [
        ModelArch(N_MAX_LAYERS, LayersConfig(1/2, 1, 1/2), LayersConfig(width_delta, 1, width_delta), LayersConfig(nn.ReLU, nn.Identity, nn.ReLU),     LayersConfig(True,  True, True)),
        ModelArch(N_MAX_LAYERS, LayersConfig(1/2, 1, 1/4), LayersConfig(width_delta, 1, width_delta), LayersConfig(nn.ReLU, nn.Identity, nn.Identity), LayersConfig(True,  True, True)),
        ModelArch(N_MAX_LAYERS, LayersConfig(1/4, 1, 1/4), LayersConfig(width_delta, 1, width_delta), LayersConfig(nn.ReLU, nn.Identity, nn.Identity), LayersConfig(False, True, False)),
        ModelArch(N_MAX_LAYERS, LayersConfig(1/4, 1, 1/2), LayersConfig(width_delta, 1, width_delta), LayersConfig(nn.ReLU, nn.Identity, nn.ReLU),     LayersConfig(False, True, False)),
    ]

    INSERTION_TYPES = list(LayerInsertionType)
    LAYER_MODES     = list(LayerModeType)[::-1]
    MUTATION_MODES  = list(MutationMode)

    TRAIN_TARGETS = [
        (train_reserved_saecollab_tolerance_model, "sae_tolerance_model"),
        (train_baseline_dense_model,               "dense_model"),
    ]

    # -- Step 1: tabular baselines + hp, all mazes, before pool opens ------
    maze_configs = {}  # maze_path -> (maze_env, hp)
    for maze_path in MAZES:
        maze_env  = MazeEnv(maze_path, rewards_scaled=False, pass_through_walls=False)
        EPISODES  = 400
        MAX_STEPS = int(len(list(Action)) * maze_env.opens_count)
        LEARN_STEPS = 4
        if isinstance(learns_by_epochs,int):
            LEARN_STEPS = np.ceil(MAX_STEPS / learns_by_epochs)
            LEARN_STEPS = int(LEARN_STEPS)

        print(f"[INFO] {basename(maze_path)}: MAX_STEPS = {MAX_STEPS}, LEARN_STEPS = {LEARN_STEPS}")

        epsilon_decay = exp_decay_factor_to(
            final_epsilon=0.1,
            final_step=MAX_STEPS * EPISODES,
            epsilon_start=1.0,
            convergence_threshold=0.01,
        )

        hp = GlobalHyperparameters(
            learning_rate           = 1e-5,
            new_layer_learning_rate = 5e-5,
            discount_factor         = 0.999,
            epsilon_decay           = epsilon_decay,
            episodes                = EPISODES,
            max_steps               = MAX_STEPS,
            batch_size              = 512,
            steps_learn_interval    = LEARN_STEPS,
            rolling_window_size     = 20,
            insert_patience         = 15,
            insert_min_goals        = 5,
            insert_min_variance     = 0.6,
        )

        try:
            state.train_tabular_agent(maze_path, hp, runs=1)
        except Exception as e:
            print(f"[WARNING] Cant Train Tabular Agent: {e}")
        state.save_a_star_qtable(maze_path)
        maze_configs[maze_path] = (maze_env, hp)

    all_jobs = []
    for maze_path in MAZES:
        maze_env, hp = maze_configs[maze_path]
        maze_tag     = basename(maze_path).split(".")[0]

        for state_representation in STATE_REPRESENTATIONS:
            gym_env    = MazeGymWrapper(maze_env, **state_representation.opts)
            base_width = gym_env.action_size * maze_env.rows * maze_env.cols

            for arch in ARCHITECTURES:
                for insertion_type in INSERTION_TYPES:
                    concrete_arch = gen_concrete_arch(base_width, gym_env, arch, insertion_type)

                    for layer_mode in LAYER_MODES:
                        for mutation_mode in MUTATION_MODES:
                            save_dir = os.path.join(
                                dir_path, str(seed),
                                maze_tag,
                                state_representation.tag,
                                arch.tag,
                                insertion_type.tag,
                                layer_mode.tag,
                                str(mutation_mode).split(".")[-1],
                            )
                            os.makedirs(save_dir, exist_ok=True)

                            for train_fn, subdir in TRAIN_TARGETS:
                                model_dir    = os.path.join(save_dir, subdir)
                                model_path   = os.path.join(model_dir, "model.pth")
                                metrics_path = os.path.join(model_dir, "metrics.csv")
                                os.makedirs(model_dir, exist_ok=True)

                                if os.path.exists(model_path):
                                    continue

                                all_jobs.append({
                                    "maze_path"    : maze_path,
                                    "model_path"   : model_path,
                                    "metrics_path" : metrics_path,
                                    "train_fn"     : train_fn.__name__,
                                    "train_tag"    : subdir,
                                    "hp"           : hp,
                                    "state_repr"   : state_representation,
                                    "concrete_arch": concrete_arch,
                                    "model_arch"   : arch,
                                    "mode_type"    : layer_mode.tag,
                                    "mutation_mode": mutation_mode,
                                    "runs"         : 1,
                                    "verbose"      : False,
                                    "maze_wrapper" : maze_wrapper,
                                    "insertion_type" : insertion_type
                                })

    total = len(all_jobs)

    # -- Progress helpers --------------------------------------------------
    WINDOW       = 50
    finish_times: deque = deque(maxlen=WINDOW)

    def fmt_time(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        if h:
            return f"{h}h {m:02d}m {s:02d}s"
        if m:
            return f"{m}m {s:02d}s"
        return f"{s}s"

    done       = 0
    errors     = 0
    start_time = time.time()

    def print_progress(job: dict, job_elapsed: float, error: Exception | None = None):
        nonlocal done, errors
        finish_times.append(time.time())

        if len(finish_times) >= 2:
            window_span = finish_times[-1] - finish_times[0]
            rate        = (len(finish_times) - 1) / window_span if window_span > 0 else 0
            remaining   = total - done
            eta_str     = fmt_time(remaining / rate) if rate > 0 else "?"
            rate_str    = f"{rate * 60:.1f} jobs/min"
        else:
            eta_str  = "calculating..."
            rate_str = "\u2014"

        pct      = done / total if total else 0
        bar_fill = int(40 * pct)
        bar      = "\u2588" * bar_fill + "\u2591" * (40 - bar_fill)
        status   = "\u2713" if error is None else "\u2717"
        ts       = time.strftime("%H:%M:%S")

        print(
            f"[{ts}] {status} [{done:>5}/{total}] ({pct*100:5.1f}%) "
            f"| took {fmt_time(job_elapsed):<9}| ETA {eta_str} @ {rate_str}"
        )
        if error:
            print(f"  \u26a0 {type(error).__name__}: {str(error)[:100]}")

        if done == 1 or done % 10 == 0 or done == total:
            sep = "\u2500" * 72
            print(sep)
            print(f"  Done: {done}/{total}  |  Errors: {errors}")
            print(f"  [{bar}] {pct*100:.1f}%  |  ETA: {eta_str}  |  {rate_str}")
            print(f"  Last: {job['model_path']}")
            print(sep)

        sys.stdout.flush()

    print(f"\n[INFO] Jobs to run : {total}")
    print(f"[INFO] Workers     : {max_workers}")
    print(f"[INFO] Start       : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] Pre-warming {max_workers} workers (CUDA init + imports)...")
    print("=" * 72)

    # -- Step 3: submit everything at once, iterate completions immediately -
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_warm_worker) as pool:
        futures = {pool.submit(train_thread, **job): (job, time.time()) for job in all_jobs}
        print(f"[INFO] All {total} jobs submitted \u2014 workers running...\n")

        for future in as_completed(futures):
            job, submit_time = futures[future]
            done += 1
            error = None
            try:
                future.result()
            except Exception as e:
                error = e
                errors += 1
                print("\n----- WORKER ERROR -----")
                print("JOB:", job["model_path"])
                print(traceback.format_exc())
            print_progress(job, time.time() - submit_time, error)

    total_time = time.time() - start_time
    print("\n" + "=" * 72)
    print(f"  ALL JOBS COMPLETE")
    print(f"  Total jobs : {total}")
    print(f"  Successful : {total - errors}")
    print(f"  Errors     : {errors}")
    print(f"  Wall time  : {fmt_time(total_time)}")
    print(f"  Avg/job    : {fmt_time(total_time / total) if total else '\u2014'}")
    print("=" * 72)

    import shutil
    print(f"[INFO] Deleting Cache Folder: {_TEMP_BASELINE_CACHE_DIR}")
    try:
        shutil.rmtree(_TEMP_BASELINE_CACHE_DIR)
    except Exception as e:
        print(f"[ERROR] {e}")


def fast_experiment_2(
    dir_path: str = None,
    seed=None,
    TABULAR_QLEARNING_PATH="./c_qlearning/build/agentTrain.exe",
    max_workers=6,
    learns_by_epochs=None,
    maze_wrapper=GPUMazeWrapper,
    early_stop_episodes=60,
    **kwargs,
):
    """
    Optimized experiment runner over fast_experiment_1:
      - Baseline deduplication: baselines depend on (maze, repr, arch, insertion_type)
        but NOT on (layer_mode, mutation_mode). Trains each unique baseline once,
        then copies results to all duplicate locations. Cuts ~46% of total jobs.
      - Early stopping enabled by default (100% success for 60 consecutive episodes).
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from collections import deque
    import traceback
    import shutil

    dir_path = dir_path or "experiment_1"
    seed = seed or 333
    set_seed(seed)

    state = AblationProgramState.load_from_json(dir_path, seed)
    if state is None:
        state = AblationProgramState(TABULAR_QLEARNING_PATH, dir_path, seed)
        state.env_update()

    # -- Experiment configuration (same as fast_experiment_1) ---------------
    MAZES = [
        "./mazes/big_eg.maze",
        "./mazes/medium_eg.maze",
        "./mazes/small_eg.maze",
    ]

    STATE_REPRESENTATIONS = [
        StateRepresentation(state_encoder=StateEncoder.ONE_HOT),
        StateRepresentation(state_encoder=StateEncoder.COORDS, possible_actions_feature=True),
        StateRepresentation(state_encoder=StateEncoder.COORDS_NORM, num_last_states=2, visited_count=True),
        StateRepresentation(state_encoder=StateEncoder.ONE_HOT, possible_actions_feature=True),
    ]


    '''
    Tabela Q = R * C * A
    Arquitetura Rede Neural por camada = (R * C * A) / K


    '''

    N_MAX_LAYERS = 4
    width_delta = 1 / N_MAX_LAYERS
    ARCHITECTURES = [
        ModelArch(N_MAX_LAYERS, LayersConfig(1/2, 1, 1/2), LayersConfig(width_delta, 1, width_delta), LayersConfig(nn.ReLU, nn.Identity, nn.ReLU),     LayersConfig(True,  True, True)),
        ModelArch(N_MAX_LAYERS, LayersConfig(1/2, 1, 1/4), LayersConfig(width_delta, 1, width_delta), LayersConfig(nn.ReLU, nn.Identity, nn.Identity), LayersConfig(True,  True, True)),
        ModelArch(N_MAX_LAYERS, LayersConfig(1/4, 1, 1/4), LayersConfig(width_delta, 1, width_delta), LayersConfig(nn.ReLU, nn.Identity, nn.Identity), LayersConfig(False, True, False)),
        ModelArch(N_MAX_LAYERS, LayersConfig(1/4, 1, 1/2), LayersConfig(width_delta, 1, width_delta), LayersConfig(nn.ReLU, nn.Identity, nn.ReLU),     LayersConfig(False, True, False)),
    ]

    INSERTION_TYPES = list(LayerInsertionType)
    LAYER_MODES     = list(LayerModeType)[::-1]
    MUTATION_MODES  = list(MutationMode)

    # -- Step 1: tabular baselines + hp per maze ---------------------------
    maze_configs = {}
    for maze_path in MAZES:
        maze_env  = MazeEnv(maze_path, rewards_scaled=False, pass_through_walls=False)
        EPISODES  = 400
        MAX_STEPS = int(len(list(Action)) * maze_env.opens_count)
        LEARN_STEPS = 4
        if isinstance(learns_by_epochs, int):
            LEARN_STEPS = int(np.ceil(MAX_STEPS / learns_by_epochs))

        print(f"[INFO] {basename(maze_path)}: MAX_STEPS = {MAX_STEPS}, LEARN_STEPS = {LEARN_STEPS}")

        epsilon_decay = exp_decay_factor_to(
            final_epsilon=0.1,
            final_step=MAX_STEPS * EPISODES,
            epsilon_start=1.0,
            convergence_threshold=0.01,
        )

        hp = GlobalHyperparameters(
            learning_rate           = 1e-5,
            new_layer_learning_rate = 5e-5,
            discount_factor         = 0.999,
            epsilon_decay           = epsilon_decay,
            episodes                = EPISODES,
            max_steps               = MAX_STEPS,
            batch_size              = 1024,
            steps_learn_interval    = LEARN_STEPS,
            rolling_window_size     = 20,
            insert_patience         = 15,
            insert_min_goals        = 5,
            insert_min_variance     = 0.6,
        )

        try:
            state.train_tabular_agent(maze_path, hp, runs=1)
        except Exception as e:
            print(f"[WARNING] Cant Train Tabular Agent: {e}")
        state.save_a_star_qtable(maze_path)
        maze_configs[maze_path] = (maze_env, hp)

    # -- Step 2: build job lists with baseline deduplication ----------------
    sae_jobs = []
    baseline_canonical = {}
    baseline_copies = {}

    for maze_path in MAZES:
        maze_env, hp = maze_configs[maze_path]
        maze_tag     = basename(maze_path).split(".")[0]

        for state_representation in STATE_REPRESENTATIONS:
            gym_env    = MazeGymWrapper(maze_env, **state_representation.opts)
            base_width = gym_env.action_size * maze_env.rows * maze_env.cols

            for arch in ARCHITECTURES:
                for insertion_type in INSERTION_TYPES:
                    concrete_arch = gen_concrete_arch(base_width, gym_env, arch, insertion_type)
                    baseline_key = (maze_tag, state_representation.tag, arch.tag, insertion_type.tag)

                    for layer_mode in LAYER_MODES:
                        for mutation_mode in MUTATION_MODES:
                            save_dir = os.path.join(
                                dir_path, str(seed),
                                maze_tag,
                                state_representation.tag,
                                arch.tag,
                                insertion_type.tag,
                                layer_mode.tag,
                                str(mutation_mode).split(".")[-1],
                            )
                            os.makedirs(save_dir, exist_ok=True)

                            # -- SAE job (always unique per full combo) ----
                            sae_dir      = os.path.join(save_dir, "sae_tolerance_model")
                            sae_model    = os.path.join(sae_dir, "model.pth")
                            sae_metrics  = os.path.join(sae_dir, "metrics.csv")
                            os.makedirs(sae_dir, exist_ok=True)

                            if not os.path.exists(sae_model):
                                sae_jobs.append({
                                    "maze_path"          : maze_path,
                                    "model_path"         : sae_model,
                                    "metrics_path"       : sae_metrics,
                                    "train_fn"           : "train_reserved_saecollab_tolerance_model",
                                    "train_tag"          : "SAE Tolerance",
                                    "hp"                 : hp,
                                    "state_repr"         : state_representation,
                                    "concrete_arch"      : concrete_arch,
                                    "model_arch"         : arch,
                                    "mode_type"          : layer_mode.tag,
                                    "mutation_mode"      : mutation_mode,
                                    "runs"               : 1,
                                    "verbose"            : False,
                                    "maze_wrapper"       : maze_wrapper,
                                    "early_stop_episodes": early_stop_episodes,
                                })

                            # -- Baseline job (deduplicated) ---------------
                            dense_dir    = os.path.join(save_dir, "dense_model")
                            dense_model  = os.path.join(dense_dir, "model.pth")
                            dense_metrics= os.path.join(dense_dir, "metrics.csv")
                            os.makedirs(dense_dir, exist_ok=True)

                            if os.path.exists(dense_model):
                                continue  # already trained from previous run

                            if baseline_key not in baseline_canonical:
                                # First occurrence -> this is the canonical training job
                                baseline_canonical[baseline_key] = {
                                    "maze_path"          : maze_path,
                                    "model_path"         : dense_model,
                                    "metrics_path"       : dense_metrics,
                                    "train_fn"           : "train_baseline_dense_model",
                                    "train_tag"          : "Baseline",
                                    "hp"                 : hp,
                                    "state_repr"         : state_representation,
                                    "concrete_arch"      : concrete_arch,
                                    "model_arch"         : arch,
                                    "mode_type"          : layer_mode.tag,
                                    "mutation_mode"      : mutation_mode,
                                    "runs"               : 1,
                                    "verbose"            : False,
                                    "maze_wrapper"       : maze_wrapper,
                                    "early_stop_episodes": early_stop_episodes,
                                }
                                baseline_copies[baseline_key] = []
                            else:
                                # Duplicate -> just record where to copy results
                                baseline_copies[baseline_key].append(dense_dir)

    baseline_jobs = list(baseline_canonical.values())
    all_jobs = sae_jobs + baseline_jobs

    n_sae      = len(sae_jobs)
    n_baseline = len(baseline_jobs)
    n_deduped  = sum(len(v) for v in baseline_copies.values())
    total      = len(all_jobs)

    print(f"\n[INFO] SAE jobs        : {n_sae}")
    print(f"[INFO] Baseline jobs     : {n_baseline}")
    print(f"[INFO] Total jobs        : {total}")
    print(f"[INFO] Early stop        : {'off' if early_stop_episodes == 0 else f'{early_stop_episodes} ep @ 100%'}")
    print(f"[INFO] Workers           : {max_workers}")
    print(f"[INFO] Start             : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] Pre-warming {max_workers} workers (CUDA init + imports)...")
    print("=" * 72)

    # -- Progress helpers --------------------------------------------------
    WINDOW       = 50
    finish_times: deque = deque(maxlen=WINDOW)

    def fmt_time(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        if h:
            return f"{h}h {m:02d}m {s:02d}s"
        if m:
            return f"{m}m {s:02d}s"
        return f"{s}s"

    done       = 0
    errors     = 0
    start_time = time.time()

    def print_progress(job: dict, job_elapsed: float, error=None):
        nonlocal done, errors
        finish_times.append(time.time())

        if len(finish_times) >= 2:
            window_span = finish_times[-1] - finish_times[0]
            rate        = (len(finish_times) - 1) / window_span if window_span > 0 else 0
            remaining   = total - done
            eta_str     = fmt_time(remaining / rate) if rate > 0 else "?"
            rate_str    = f"{rate * 60:.1f} jobs/min"
        else:
            eta_str  = "calculating..."
            rate_str = "\u2014"

        pct      = done / total if total else 0
        bar_fill = int(40 * pct)
        bar      = "\u2588" * bar_fill + "\u2591" * (40 - bar_fill)
        status   = "\u2713" if error is None else "\u2717"
        ts       = time.strftime("%H:%M:%S")

        print(
            f"[{ts}] {status} [{done:>5}/{total}] ({pct*100:5.1f}%) "
            f"| took {fmt_time(job_elapsed):<9}| ETA {eta_str} @ {rate_str}"
        )
        if error:
            print(f"  \u26a0 {type(error).__name__}: {str(error)[:100]}")

        if done == 1 or done % 10 == 0 or done == total:
            sep = "\u2500" * 72
            print(sep)
            print(f"  Done: {done}/{total}  |  Errors: {errors}")
            print(f"  [{bar}] {pct*100:.1f}%  |  ETA: {eta_str}  |  {rate_str}")
            print(f"  Last: {job['model_path']}")
            print(sep)

        sys.stdout.flush()

    # -- Step 3: submit & run ----------------------------------------------
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_warm_worker) as pool:
        futures = {pool.submit(train_thread, **job): (job, time.time()) for job in all_jobs}
        print(f"[INFO] All {total} jobs submitted \u2014 workers running...\n")

        for future in as_completed(futures):
            job, submit_time = futures[future]
            done += 1
            error = None
            try:
                future.result()
            except Exception as e:
                error = e
                errors += 1
                print("\n----- WORKER ERROR -----")
                print("JOB:", job["model_path"])
                print(traceback.format_exc())
            print_progress(job, time.time() - submit_time, error)

    # -- Step 4: copy deduplicated baseline results ------------------------
    copies_done = 0
    copies_failed = 0
    for baseline_key, copy_dirs in baseline_copies.items():
        canonical = baseline_canonical[baseline_key]
        src_model   = canonical["model_path"]
        src_metrics = canonical["metrics_path"]

        if not os.path.exists(src_model):
            copies_failed += len(copy_dirs)
            continue

        for dst_dir in copy_dirs:
            try:
                dst_model   = os.path.join(dst_dir, "model.pth")
                dst_metrics = os.path.join(dst_dir, "metrics.csv")
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy2(src_model, dst_model)
                if os.path.exists(src_metrics):
                    shutil.copy2(src_metrics, dst_metrics)
                copies_done += 1
            except Exception as e:
                copies_failed += 1
                print(f"[WARNING] Copy failed for {dst_dir}: {e}")

    total_time = time.time() - start_time
    print("\n" + "=" * 72)
    print(f"  ALL JOBS COMPLETE")
    print(f"  Total trained  : {total}")
    print(f"  Successful     : {total - errors}")
    print(f"  Errors         : {errors}")
    print(f"  Baseline copies: {copies_done} ok, {copies_failed} failed")
    print(f"  Wall time      : {fmt_time(total_time)}")
    print(f"  Avg/job        : {fmt_time(total_time / total) if total else '\u2014'}")
    print("=" * 72)


SELECTABLE_EXPERIMENTS = [experiment_1, fast_experiment_1, fast_experiment_2]
