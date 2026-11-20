"""`python -m marketcast ...` entry point.

The __name__ guard matters on Windows: the panel builder uses worker processes, and each spawned worker
re-imports this module (as "__mp_main__"). Without the guard every worker would start the CLI again.
"""
from marketcast.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
