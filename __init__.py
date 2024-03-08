import sys
from colorama import Fore

arg = sys.argv[1] if len(sys.argv) > 1 else None

# Use match statement
match arg:
  case 'kindistry':
    print("Kindistry")

  case 'eev':
    from eev.env import run as eev_run
    print("eev")
    eev_run()

  case _:
    print(Fore.RED, 'Invalid argument: ', Fore.RESET, arg)
