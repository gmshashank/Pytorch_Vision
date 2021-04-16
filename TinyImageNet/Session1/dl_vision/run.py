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

    parser.add_argument(
        "-m", "--custom_model", default=None, type=bool, help="Enable Custom Model"
    )

    # parse arguments
    args = parser.parse_args()

    # load the config file
    config = load_config(args.config)

    # # iterating the parser arguments
    # for arg in vars(args):
    #     print(arg, getattr(args, arg))
    # print(getattr(args, "custom_model"))

    # create a runner
    runner = runners.Runner(config, custom_model=getattr(args, "custom_model"))

    # setup train parameters
    runner.setup_train()

    # print model summary
    print("model_summary")
    runner.model_summary(input_size=(3, 32, 32))

    # Find LR
    runner.find_lr()

    # Training the model
    # runner.trainer.train()
    runner.train_lr(use_best_lr=True)

    # plot the metrics
    plt = runner.plot_metrics()

    # Saving the plot to file
    plt.savefig("Metrics.png")

    # plot GradCAM
    target_layers = ["layer1", "layer2", "layer3", "layer4"]
    runner.plot_gradcam(target_layers=target_layers)

    # plot misclassified
    runner.plot_misclassified(target_layers=target_layers)

    print("done")


if __name__ == "__main__":
    main()
    # python run.py --config=config.yml --custom_model=True
