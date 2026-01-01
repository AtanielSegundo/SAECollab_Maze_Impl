import time

from os.path import basename
from Ablation import AblationProgramState,GlobalHyperparameters
from models.QModels import exp_decay_factor_to

from env.MazeEnv import *
from env.MazeWrapper import StateEncoder, MazeGymWrapper


def experiment_1(dir_path:str=None,seed=None):
    dir_path = dir_path or 'experiment_1'
    seed     = seed or 333
    TABULAR_QLEARNING_PATH = "./c_qlearning/build/agentTrain.exe"
    
    state = AblationProgramState.load_from_json(dir_path,seed)
    
    if state is None:
        state = AblationProgramState(
            TABULAR_QLEARNING_PATH,
            dir_path,
            seed
        )
        state.env_update()

    # COMBINATORIAL OPTIONS START
    MAZES = ["./mazes/small_eg.maze","./mazes/medium_eg.maze","./mazes/big_eg.maze"]
        
    # COMBINATORIAL OPTIONS END

    for maze_path in MAZES[state.getLastCombIndex():]: 
        maze_tag = basename(maze_path).split(".")[0]
        state.add_save_path_head(maze_tag)   
        print(state.save_dir_path)

        maze_env = MazeEnv(maze_path, rewards_scaled=False, pass_through_walls=False)
        
        # GLOBAL HYPERPARAMETERS
        EPISODES  = 200
        MAX_STEPS =maze_env.opens_count * len(list(Action))
        
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
        
        time.sleep(4)

        state.remove_save_path_head()

        
    
SELECTABLE_EXPERIMENTS = [experiment_1] 