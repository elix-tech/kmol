from argparse import ArgumentParser

from .core.config import ScriptConfig
from .script import ScriptLauncher


def main():
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    ScriptLauncher(config=ScriptConfig.from_file(args.config)).run()

if __name__ == "__main__":
    main()