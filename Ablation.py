#!python3 Ablation.py 

import os
import json
import csv
import subprocess
import multiprocessing
from typing import *

import torch
import numpy as np
import random

from env.MazeEnv     import *
from StackedCollab.collabNet import LayersConfig,MutationMode
from env.MazeWrapper import StateEncoder, MazeGymWrapper
from models.AStar    import AStarQModel
from models.QTable   import genearate_qtable_from_model

class GlobalHyperparameters:
    def __init__(self,
                 learning_rate,
                 discount_factor,
                 epsilon_decay,
                 episodes,
                 max_steps,
                 batch_size,
                 steps_learn_interval,
                 rolling_window_size
                 ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_decay   = epsilon_decay
        self.episodes        = episodes
        self.max_steps       = max_steps
        self.batch_size      = batch_size
        self.steps_learn_interval = steps_learn_interval
        self.rolling_window_size  = rolling_window_size

class LayerInsertionType(Enum):
    CNT = "CNT" 
    CRT = "CRT" 
    DRT = "DRT" 
    ALT = "ALT"
    
    @property
    def tag(self):
        return self.value

class LayerMode:
    def __init__(self,
                 is_k_trainable:bool,
                 use_extra_branch:bool
                ):
        self.is_k_trainable   = is_k_trainable
        self.use_extra_branch = use_extra_branch

class LayerModeType(Enum):
    M1 = LayerMode(False, False)
    M2 = LayerMode(False, True)
    M3 = LayerMode(True , False)
    M4 = LayerMode(True , True)

    @property
    def tag(self):
        return str(self).split(".")[-1]

class ModelArch:
    def __init__(self,
                 max_layers             : int,
                 width_multipliers      : LayersConfig,   # THE OUT LAYER MULTIPLIER SHOULD BE 1
                 delta_width_multipliers: LayersConfig,
                 activation             : LayersConfig,
                 use_bias               : LayersConfig
                 )         : 
        self.max_layers              = max_layers
        self.width_multipliers       = width_multipliers
        self.delta_width_multipliers = delta_width_multipliers
        self.activation              = activation
        self.use_bias                = use_bias
    
    @property
    def tag(self):
        def GAN(act) -> str:
            #GET ACTIVATION NAME
            return str(act).split(".")[-1].split("'")[0]
        def GUB(b:bool) -> str:
            return "T" if b else "F"

        return f"{self.max_layers}" \
               f"_H{self.width_multipliers.hidden}O{self.width_multipliers.out}E{self.width_multipliers.extra}" \
               f"_H{GAN(self.activation.hidden)}O{GAN(self.activation.out)}E{GAN(self.activation.extra)}" \
               f"_H{GUB(self.use_bias.hidden)}O{GUB(self.use_bias.out)}E{GUB(self.use_bias.extra)}"


class StateRepresentation:
    NUM_LAST_STATES_TAG          = "nls"
    NUM_LAST_ACTIONS_TAG         = "nla"
    POSSIBLE_ACTIONS_FEATURE_TAG = "paf"
    VISITED_COUNT_TAG            = "vcnt"

    def __init__(self,
        state_encoder           :StateEncoder  = StateEncoder.COORDS,
        num_last_states         :int           = None,  
        num_last_actions        :int           = None, 
        possible_actions_feature:bool          = False, 
        visited_count           :bool          = False,
    ) :
        self.opts = {
            "state_encoder"           : state_encoder,
            "num_last_states"         : num_last_states,
            "num_last_actions"        : num_last_actions,
            "possible_actions_feature": possible_actions_feature,
            "visited_count"           : visited_count
        }
    
    @property
    def tag(self):
        tag = ""
        tag += str(self.opts["state_encoder"]).split(".")[-1]
        if not self.opts["num_last_states"] is None:
            tag += f"_{StateRepresentation.NUM_LAST_STATES_TAG}_" + str(self.opts["num_last_states"])
        if not self.opts["num_last_actions"] is None:
            tag += f"_{StateRepresentation.NUM_LAST_ACTIONS_TAG}_" + str(self.opts["num_last_actions"])
        if self.opts["possible_actions_feature"]:
            tag += f"_{StateRepresentation.POSSIBLE_ACTIONS_FEATURE_TAG}"
        if self.opts["visited_count"]:
            tag += f"_{StateRepresentation.VISITED_COUNT_TAG}"
        return tag

class ModelTrainMetrics:
    def __init__(self,
                 episode         :List[int]   = None,
                 reward          :List[float] = None,
                 cumulative_goals:List[int]   = None,
                 sucess_rate     :List[float] = None,
                 loss            :List[float] = None,
                 steps           :List[int]   = None,
                 parameters_cnt  :List[int]   = None      
                 ):
        self.episode          :List[int]   = episode          or list()
        self.reward           :List[float] = reward           or list()
        self.cumulative_goals :List[int]   = cumulative_goals or list()
        self.sucess_rate      :List[float] = sucess_rate      or list()
        self.loss             :List[float] = loss             or list()
        self.steps            :List[int]   = steps            or list()
        self.parameters_cnt   :List[int]   = parameters_cnt   or list()

    def __len__(self):
        return min(map(len,[self.episode,self.reward,self.cumulative_goals,
                            self.sucess_rate,self.loss,self.steps,
                            self.parameters_cnt]))
    
    def append(self,episode,reward,cumulative_goals,sucess_rate,loss,steps,parameters_cnt):
        self.episode          = episode
        self.reward           = reward
        self.cumulative_goals = cumulative_goals
        self.sucess_rate      = sucess_rate
        self.loss             = loss
        self.steps            = steps
        self.parameters_cnt   = parameters_cnt

    def save(self,path:str):
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            header_row_str = "episode,reward,cumulative_goals,success_rate,training_loss,steps,parameters"
            writer.writerow(header_row_str.split(","))
            for idx in range(len(self)):
                writer.writerow(f"{self.episode[idx]},{self.reward[idx]},{self.cumulative_goals[idx]},{self.success_rate[idx]},{self.training_loss[idx]},{self.steps[idx]},{self.parameters_cnt[idx]}")

class AblationProgramState:
    STATE_JSON_NAME   = ".ablation_state.json"
    METRICS_FILE_NAME = "metrics.csv"
    QTABLE_FILE_NAME  = "tabular.qtable"

    def __init__(self,
                 tabular_trainer_path:str          = None,
                 save_dir_path       :str          = "./Ablation",
                 initial_seed                      = 67,
                 extended_info       : dict        = None,
                 completed_iteration : int         = -1
    ):
        self.tabular_trainer_path = tabular_trainer_path
        self.seed                 = initial_seed
        self.comb_ptr             = -1
        self.comb_tags            = []
        self.save_dir_path        = save_dir_path
        self.save_dir_path        = os.path.join(self.save_dir_path,str(self.seed))
        self.program_state_path   = os.path.join(self.save_dir_path,AblationProgramState.STATE_JSON_NAME)
        self.extended             = extended_info or dict()
        self.completed_iteration  = completed_iteration

    def iteration_to_indices(self, iteration: int, dimensions: list) -> list:
        """Convert linear iteration number to multi-dimensional indices"""
        indices = []
        remaining = iteration
        
        # Work from innermost to outermost dimension
        for i in range(len(dimensions) - 1, -1, -1):
            indices.insert(0, remaining % dimensions[i])
            remaining //= dimensions[i]
        
        return indices

    def get_skip_indices(self, dimensions: list) -> list:
        """Get the starting indices for each dimension based on completed_iteration"""
        if self.completed_iteration < 0:
            return [0] * len(dimensions)
        
        # Get indices of next iteration to run
        next_iteration = self.completed_iteration + 1
        return self.iteration_to_indices(next_iteration, dimensions)

    def env_update(self):
        os.makedirs(self.save_dir_path,exist_ok=True)
        self.save_json()

    def save_json(self):
        data = {
            "tabular_trainer_path": self.tabular_trainer_path,
            "save_dir_path"       : self.save_dir_path,
            "program_state_path"  : self.program_state_path,
            "seed"                : self.seed,
            "comb_tags"           : list(map(list,self.comb_tags)),
            "extended"            : self.extended,
            "completed_iteration" : self.completed_iteration
        }
        with open(self.program_state_path, 'w') as f:
            json.dump(data,f, indent = 4, ensure_ascii = False)

    def update_comb_tags(self,tag:str):
        if len(self.comb_tags) < self.comb_ptr + 1:
            self.comb_tags.append(set())

        self.comb_tags[self.comb_ptr].add(tag)

    def add_save_path_head(self,p:str):
        self.comb_ptr += 1
        self.update_comb_tags(p)
        self.save_dir_path = os.path.join(self.save_dir_path,p)
        os.makedirs(self.save_dir_path,exist_ok=True)

    def remove_save_path_head(self):
        self.comb_ptr -= 1
        self.save_dir_path = os.path.dirname(self.save_dir_path)

    def mark_iteration_complete(self, iteration_num: int):
        """Mark an iteration as complete and save state"""
        self.completed_iteration = iteration_num
        self.save_json()
    
    def should_skip_iteration(self, iteration_num: int) -> bool:
        """Check if this iteration was already completed"""
        return iteration_num <= self.completed_iteration
    
    def train_tabular_agent(self,maze_path:str,hp:GlobalHyperparameters,
                            repetitions:int=1):        
        if repetitions == 1:
            tabular_save_path = os.path.join(self.save_dir_path,"tabular")
            os.makedirs(tabular_save_path,exist_ok=True)
            subprocess.run(
                [self.tabular_trainer_path, 
                maze_path,
                "--lr", str(hp.learning_rate),
                "--df", str(hp.discount_factor),
                "--decay", f"{hp.epsilon_decay}",
                "--episodes", str(hp.episodes),
                "--max_steps", str(hp.max_steps),
                "--seed", str(self.seed),
                "--qtable_path", f"{os.path.join(tabular_save_path,AblationProgramState.QTABLE_FILE_NAME)}",
                "--metrics_path", f"{os.path.join(tabular_save_path,AblationProgramState.METRICS_FILE_NAME)}",
                ],
                check=False,
                stdout=subprocess.DEVNULL
            )
        elif repetitions > 1 :
            tabular_processes = []
            for repetition in range(repetitions):
                tabular_save_path = os.path.join(self.save_dir_path,f"tabular_{repetition}")
                os.makedirs(tabular_save_path,exist_ok=True)
                seed = self.seed + repetition
                qtable_path  = os.path.join(tabular_save_path,AblationProgramState.QTABLE_FILE_NAME)
                metrics_path = os.path.join(tabular_save_path,AblationProgramState.METRICS_FILE_NAME)
                p = multiprocessing.Process(
                    target=subprocess.run,
                    args=(
                        [
                        self.tabular_trainer_path, 
                        maze_path,
                        "--lr", str(hp.learning_rate),
                        "--df", str(hp.discount_factor),
                        "--decay", f"{hp.epsilon_decay}",
                        "--episodes", str(hp.episodes),
                        "--max_steps", str(hp.max_steps),
                        "--seed", str(seed),
                        "--qtable_path", f"{qtable_path}",
                        "--metrics_path", f"{metrics_path}",
                        ],
                    ),
                    kwargs={"check":False,"stdout":subprocess.DEVNULL}
                )
                p.start()
                tabular_processes.append(p)

            for idx,process in enumerate(tabular_processes):
                process.join()
                print(f"[INFO] Tabular Process {idx+1} Ended")

        else:
            print(f"[ERROR] Invalid Number of Repetitions {repetitions} For Tabular Agent")
            exit(-1)

    def get_current_iteration(self, indices: list, dimensions: list) -> int:
        """
        Convert multi-dimensional indices to linear iteration number.
        
        Args:
            indices: Current position in each dimension
            dimensions: Size of each dimension
        
        Returns:
            Linear iteration number (0-based)
        """
        current_iter = 0
        multiplier = 1
        
        # Start from innermost dimension (last) to outermost (first)
        for i in range(len(dimensions) - 1, -1, -1):
            current_iter += indices[i] * multiplier
            multiplier *= dimensions[i]
        
        return current_iter
    
    def save_a_star_qtable(self,maze_path:str):
        env = MazeGymWrapper(MazeEnv(maze_path))
        a_star_model = AStarQModel(env)
        qtable =  genearate_qtable_from_model(
            env, a_star_model, env.action_size,
            get_q_val_method='__call__',
            use_extras=False
        )
        qtable.save(os.path.join(self.save_dir_path,"astar.qtable"))

    @classmethod
    def load_from_json(cls, dir_path, seed) -> Optional[Self]:
        experiment_path = os.path.join(dir_path, str(seed))
        if not os.path.exists(experiment_path): 
            return None
        
        state_path = os.path.join(experiment_path, AblationProgramState.STATE_JSON_NAME)
        if not os.path.exists(state_path): 
            return None
        
        with open(state_path, 'r') as f:
            data = json.load(f)
        
        if data == None: return None

        ablation_state = AblationProgramState()
        ablation_state.tabular_trainer_path = data["tabular_trainer_path"]
        ablation_state.program_state_path   = data["program_state_path"]
        ablation_state.seed                 = data["seed"]
        ablation_state.extended             = data["extended"]
        ablation_state.completed_iteration  = data.get("completed_iteration", -1)
        
        # Set save_dir_path to base experiment directory
        ablation_state.save_dir_path = experiment_path

        return ablation_state
    

def set_seed(seed):
    """Set seeds for reproducibility across numpy, random, torch."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass