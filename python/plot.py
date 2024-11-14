import argparse
from pathlib import Path
import os

import isaaclab.isaaclab_configuration
from utils.plotter import Plotter


def main() -> None:
    """Command line interface. change that is ok"""
    parser = argparse.ArgumentParser()
    parser.add_argument("policy_file_path", type=Path)
    parser.add_argument("pickle_prefix", type=str)
    parser.add_argument("--history", type=int, default=1)
    parser.add_argument("--estimate", action="store_true", default=False)
    options = parser.parse_args()

    conf_file = isaaclab.isaaclab_configuration.detect_config_file(options.policy_file_path)
    config = isaaclab.isaaclab_configuration.load_configuration(conf_file)

    policy_abs_path = os.path.abspath(options.policy_file_path)
    re = f"{policy_abs_path}/logs/{options.pickle_prefix}*.pkl"
    plotter = Plotter(config, re, options.history, options.estimate)
    plotter.merge_data()
    plotter.plot(None)


if __name__ == "__main__":
    main()
