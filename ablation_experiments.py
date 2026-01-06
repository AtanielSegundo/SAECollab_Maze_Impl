import time

from os.path import basename
from Ablation import AblationProgramState,GlobalHyperparameters, \
                     StateRepresentation
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

    # COMBINATORIAL OPTIONS END

    for maze_path in MAZES[state.getLastCombIndex():]: 
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

        for state_representation in STATE_REPRESENTATIONS[state.getLastCombIndex():]:
            repr_tag = state_representation.tag
            state.add_save_path_head(repr_tag)   
            
            

            state.remove_save_path_head()

        state.remove_save_path_head()

        

SELECTABLE_EXPERIMENTS = [experiment_1] 