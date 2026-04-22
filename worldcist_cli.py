from sys              import argv
from worldcist.multi  import main as multi_main
from worldcist.single import main as single_main


if __name__ == "__main__":
    args = argv[1:]
    cli_name = argv[0]

    if any(flag in argv for flag in ('-h', '--help')) or len(args) < 1:
        print(f"Usage: python3 {cli_name} [--single | --multi]")
        exit(-1)

    if "--single" in args:
        single_main()
        
    if "--multi" in args:
        multi_main()