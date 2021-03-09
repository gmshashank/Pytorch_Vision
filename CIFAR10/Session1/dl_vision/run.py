import argparse
import os
import runner.runner as runners
import torch
import torch.optim as optim
from typing import Any, List, Tuple, Dict
from utils.config import load_config
from utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    print("main")
    parser = argparse.ArgumentParser(description="dl_vision")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )

    # parse arguments
    args = parser.parse_args()

    # load the config file
    config = load_config(args.config)

    # create a runner
    runner = runners.Runner(config)

    # setup train parameters
    runner.setup_train()

    # print model summary
    # runner.model_summary(input_size=(1, 28, 28))
    runner.model_summary(input_size=(3, 32, 32))

    # Training the model
    runner.trainer.train()

    # plot the metrics
    plt = runner.plot_metrics()

    # Saving the plot to file
    plt.savefig("Metrics.png")

    print("done")


if __name__ == "__main__":
    main()
