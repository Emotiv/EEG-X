"""Run EEG-X seeds sequentially on this machine from one YAML configuration."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="EEG-X YAML configuration")
    arguments = sys.argv[1:]
    if arguments and arguments[0] == "train":
        arguments = arguments[1:]
    args = parser.parse_args(arguments)
    from pretrain.train import run_config

    print(run_config(args.config))


if __name__ == "__main__":
    main()
