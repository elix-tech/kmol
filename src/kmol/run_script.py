from argparse import ArgumentParser

from kmol.core.config import ScriptConfig
from kmol.script import ScriptLauncher


def main():
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    ScriptLauncher(config=ScriptConfig.from_file(args.config, job_command=None)).run()


if __name__ == "__main__":
    main()
