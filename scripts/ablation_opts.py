import os
import sys
import tqdm
import shutil
from pathlib import Path

from sys import argv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ablation.traversal import traversal_search, SearchFilter

def log_search_count(search_result):
    for f_key,targets in search_result.items():
            print(f"{f_key}:")
            for t_key,val in targets.items():
                print(f"\t{t_key}:{len(val)}")

def op_in_filter_entry(search_base:str,entry:str,exclude:bool=False):
    search_targets = {
        "baseline": "dense_model/metrics.csv",
        "sae": "sae_tolerance_model/metrics.csv"
    }
    search_filters = {
        entry: SearchFilter(entry),
    }
    search_result = traversal_search(search_base, search_targets, search_filters)
    
    for k,v in search_result[entry].items():
        print(f"{k}:")
        for e in v:
            print(f"\t{e}")

    if exclude:
        for k,v in search_result[entry].items():
            for e in v:
                dir_name = os.path.dirname(e)
                model_path = os.path.join(dir_name,"model.pth")
                if os.path.exists(e):
                    os.remove(e)
                    print(f"[REMOVED] {e}")
                if os.path.exists(model_path):
                    os.remove(model_path)
                    print(f"[REMOVED] {model_path}")
 
def op_in_models(search_base:str,move_folder:str=None,delete=False):
    
    search_targets = {
        "model": "model.pth",
    }
    
    search_filters = {
        "Any": SearchFilter("model.pth"),
    }

    search_result = traversal_search(search_base, search_targets, search_filters)

    for _,v in search_result["Any"].items():
        for e in v:
            print(f"\t{e}")

    if delete:
        for _,v in search_result["Any"].items():
            for e in tqdm.tqdm(v,desc=f"Deleting Models",unit="file"):
                os.remove(e)

    if move_folder:
        if not os.path.exists(move_folder):
            print(f"[ERROR] Folder Not Found: {move_folder}")
            exit(-1)
        
        for _,v in search_result["Any"].items():
            for e in tqdm.tqdm(v,desc=f"Copying Models To {move_folder}",unit="file"):
                _save_path = os.path.join(move_folder,os.path.dirname(e).split(search_base)[-1])
                os.makedirs(_save_path,exist_ok=True)
                shutil.copy2(e,os.path.join(_save_path,os.path.basename(e)))

if __name__ == "__main__":
    if not len(argv) > 1:
        print(f"[ERROR] Expected An Search Root Dir: {argv[0]} <search base:str>")
    args = argv[1:]
    search_base = args[0]

    move_folder = None
    if "--move_folder" in args:
        idx = args.index("--move_folder")
        if len(args) < idx + 1:
            print("[ERROR] Move Folder Param Not Found")
            exit()
        move_folder = args[idx+1]
    
    opts_modes = {
       "show_big"    : lambda: op_in_filter_entry(search_base,"Big"), 
       "exclude_big" : lambda: op_in_filter_entry(search_base,"Big",True),
       "show_models" : lambda: op_in_models(search_base),
       "move_models" : lambda: op_in_models(search_base,move_folder),
       "delete_models" : lambda: op_in_models(search_base,delete=True)
    }

    for key in opts_modes.keys():
        if key in args:
            opts_modes[key]()