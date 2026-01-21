import shutil
import os
from sys import argv
from typing import Callable
from ablation_experiments import SELECTABLE_EXPERIMENTS

if __name__ == "__main__":
    
    args = argv[1:]

    PAD_SYMBOL = '='
    PAD_LEN    = 30
    CLI_PAD    = lambda : PAD_SYMBOL * PAD_LEN

    if any(flag in argv for flag in ('-h', '--help')):
        title = "AVAILABLE EXPERIMENTS"
        print(title.center(PAD_LEN * 2 + len(title), PAD_SYMBOL))
        for experiment in SELECTABLE_EXPERIMENTS :
            print(experiment.__name__)
        exit(0)

    if len(args) < 1:
        print("[ERROR] NO ARGS PROVIDED")
        exit()

    experiment_name = args[0]

    experiment = filter(lambda e: e.__name__ == experiment_name, SELECTABLE_EXPERIMENTS)
        
    try :
        experiment:Callable[[str,int],None] = list(experiment)[0]
    except:
        print("[WARNING] NO EXPERIMENT SELECTED, USING FALLBACK")
        experiment:Callable[[str,int],None] = SELECTABLE_EXPERIMENTS[0]

    print(f"[INFO] Using Experiment: {experiment.__name__}")

    custom_seed     = None
    custom_dir_path = None

    # HANDLING CUSTOM SEED
    for flag in ('-s', '--seed'):
        if flag in args:
            custom_seed = int(args[args.index(flag) + 1])
            break
    
    # HANDLING CUSTOM DIR PATH
    for flag in ('-d', '--dir'):
        if flag in args:
            custom_dir_path = args[args.index(flag) + 1]
            break

    if (flag := "--clean") in args:
        if custom_dir_path and os.path.exists(custom_dir_path):
            shutil.rmtree(custom_dir_path)

    experiment(custom_dir_path,custom_seed)