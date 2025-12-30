from sys import argv
from Ablation import AblationProgramState

def experiment_1(dir_path:str='experiment_1',seed=333):
    TABULAR_QLEARNING_PATH = "./c_qlearning/build/agentTrain.exe"
    state = AblationProgramState.load_from_json(dir_path,seed)
    
    if state is None:
        state = AblationProgramState(
            TABULAR_QLEARNING_PATH,
            dir_path,
            seed
        )
        state.env_setup()

    
SELECTABLE_EXPERIMENTS = [experiment_1] 

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

    experiment = filter(lambda e: e.__name__ == experiment_name, SELECTABLE_EXPERIMENTS)[0]

    if any(flag in args for flag in ('-s', '--seed')):
        args.index(flag)

    experiment()