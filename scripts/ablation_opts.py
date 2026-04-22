import os
from sys import argv
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ablation.traversal import traversal_search, SearchFilter

def log_search_count(search_result):
    for f_key,targets in search_result.items():
            print(f"{f_key}:")
            for t_key,val in targets.items():
                print(f"\t{t_key}:{len(val)}")

if __name__ == "__main__":
    # Removing All the Old Big Trained Maps

    search_base = "./experiment_1/24022026"

    search_targets = {
        "baseline": "dense_model/metrics.csv",
        "sae": "sae_tolerance_model/metrics.csv"
    }

    search_filters = {
        "Small" : SearchFilter("small_eg"),
        "Medium": SearchFilter("medium_eg"),
        "Big"   : SearchFilter("big_eg"),
    }

    search_result = traversal_search(search_base, search_targets, search_filters)
    
    log_search_count(search_result)

    # Show All Big mazes
    if "show_all" in argv:
        for k,v in search_result["Big"].items():
            print(f"{k}:")
            for e in v:
                print(f"\t{e}")

    if "exclude_big" in argv:
        # Exclude All Big Mazes (Models+Metrics)
        for k,v in search_result["Big"].items():
            for e in v:
                dir_name = os.path.dirname(e)
                model_path = os.path.join(dir_name,"model.pth")
                if os.path.exists(e):
                    os.remove(e)
                    print(f"[REMOVED] {e}")
                if os.path.exists(model_path):
                    os.remove(model_path)
                    print(f"[REMOVED] {model_path}")