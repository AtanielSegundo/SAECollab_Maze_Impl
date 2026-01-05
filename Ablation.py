import os
import json
import subprocess
from typing import *

METRICS_FILE_NAME = "metrics.csv"
QTABLE_FILE_NAME  = "tabular.qtable"

class GlobalHyperparameters:
    def __init__(self,learning_rate,discount_factor,epsilon_decay,episodes,max_steps):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_decay   = epsilon_decay
        self.episodes        = episodes
        self.max_steps       = max_steps

class AblationProgramState:
    STATE_JSON_NAME = ".ablation_state.json"

    def __init__(self,
                 tabular_trainer_path:str          = None,           # PATH TO TABULAR QLEARNING EXECUTABLE
                 save_dir_path       :str          = "./Ablation",
                 initial_seed                      = 67,
                 extended_info       : dict        = None,
                 comb_indexes        : list        = None
    ):
        self.tabular_trainer_path = tabular_trainer_path
        self.seed                 = initial_seed
        self.save_dir_path        = save_dir_path
        self.save_dir_path        = os.path.join(self.save_dir_path,str(self.seed))
        self.program_state_path   = os.path.join(self.save_dir_path,AblationProgramState.STATE_JSON_NAME)
        #TODO: MAYBE AN BETTER APPROACH? 
        self.comb_indexes_ptr     = -1                      
        self.comb_indexes         = comb_indexes  or list() 
        self.extended             = extended_info or dict()

    def update_comb_indexes(self):
        self.comb_indexes_ptr += 1
        if len(self.comb_indexes) < self.comb_indexes_ptr + 1 :
            self.comb_indexes.append(0)
        else:
            self.comb_indexes[self.comb_indexes_ptr] += 1

    def env_update(self):
        os.makedirs(self.save_dir_path,exist_ok=True)
        self.save_json()

    def save_json(self):
        data = {
            "tabular_trainer_path": self.tabular_trainer_path,
            "save_dir_path"       : self.save_dir_path,
            "program_state_path"  : self.program_state_path,
            "comb_indexes"        : self.comb_indexes,
            "seed"                : self.seed,
            "extended"            : self.extended
        }
        with open(self.program_state_path, 'w') as f:
            json.dump(data,f, indent = 4, ensure_ascii = False)

    def add_save_path_head(self,p:str):
        self.save_dir_path = os.path.join(self.save_dir_path,p)
        self.update_comb_indexes()
        self.env_update()

    def remove_save_path_head(self):
        self.save_dir_path = os.path.dirname(self.save_dir_path)
        self.comb_indexes_ptr -= 1 
 
    def getLastCombIndex(self):
        if len(self.comb_indexes) < self.comb_indexes_ptr+2:
            return 0
        return self.comb_indexes[self.comb_indexes_ptr+1]
    
    def train_tabular_agent(self,maze_path:str,hp:GlobalHyperparameters):        
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
             "--qtable_path", f"{os.path.join(tabular_save_path,QTABLE_FILE_NAME)}",
             "--metrics_path", f"{os.path.join(tabular_save_path,METRICS_FILE_NAME)}",
            ],
            check=False,
            stdout=subprocess.DEVNULL
        )

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
        ablation_state.save_dir_path        = data["save_dir_path"]
        ablation_state.program_state_path   = data["program_state_path"]
        ablation_state.seed                 = data["seed"]
        ablation_state.extended             = data["extended"]
        ablation_state.comb_indexes         = data["comb_indexes"]
        
        # FIX: Reset to base directory level for loop continuation
        for _ in range(len(ablation_state.comb_indexes)):
            ablation_state.save_dir_path = os.path.dirname(ablation_state.save_dir_path)

        return ablation_state