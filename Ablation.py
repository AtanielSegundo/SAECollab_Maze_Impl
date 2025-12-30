import os
import json
from typing import *

class AblationProgramState:
    STATE_JSON_NAME = ".ablation_state.json"

    def __init__(self,
                 tabular_trainer_path:str          = None,           # PATH TO TABULAR QLEARNING EXECUTABLE
                 save_dir_path       :str          = "./Ablation",
                 initial_seed                      = 67,
                 extended_info       : dict        = None
    ):
        self.tabular_trainer_path = tabular_trainer_path
        self.save_dir_path        = save_dir_path
        self.program_state_path   = os.path.join(self.save_dir_path,self.program_state_path)
        self.seed                 = initial_seed
        self.extended             = extended_info or dict()

    def env_setup(self):
        os.makedirs(os.path.join(self.save_dir_path,str(self.seed)))
        self.save()

    def save_json(self):
        data = {
            "tabular_trainer_path": self.tabular_trainer_path,
            "save_dir_path"       : self.save_dir_path,
            "program_state_path"  : self.program_state_path,
            "seed"                : self.seed,
            "extended"            : self.extended
        }
        with open(self.program_state_path, 'w') as f:
            json.dump(data,f,sort_keys = True, indent = 4, ensure_ascii = False)
    
    @classmethod
    def load_from_json(cls,dir_path) -> Optional[Self]:
        if not os.path.exists(dir_path): 
            return None
        
        state_path = os.path.join(dir_path,Self.STATE_JSON_NAME)

        if not os.path.exists(state_path): 
            return None
        
        with open(state_path,'r') as f:
            data = json.load(f)
        
        if data == None: return None

        ablation_state = AblationProgramState()
        ablation_state.tabular_trainer_path = data["tabular_trainer_path"]
        ablation_state.save_dir_path        = data["save_dir_path"]
        ablation_state.program_state_path   = data["program_state_path"]
        ablation_state.seed                 = data["seed"]
        ablation_state.extended             = data["extended"]

        return ablation_state

def experiment_1():
    pass

SELECTABLE_EXPERIMENTS = [experiment_1] 

if __name__ == "__main__":

    pass