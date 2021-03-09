import math
import torch
import torch.optim as optim
import pprint
import model.model as model_arch
import model.loss as model_loss
import data_loader.transforms as model_transforms
import data_loader.data_loaders as model_data_loaders
import utils.plot as plot

from trainer.trainer import Trainer
from torchsummary import summary

from utils.logger import setup_logger
from utils.config import get_instance, setup_seed, setup_device, setup_model_params

logger = setup_logger(__name__)


class Runner:
    def __init__(self, config):
        self.config = config

    def setup_train(self):
        config = self.config
        logger.info("Training Configuration")

        # displaying the config fie
        for line in pprint.pformat(config).split("\n"):
            logger.info(line)

        # setup seed for reproducibility of results
        setup_seed(config["seed"])

        # create model instance
        model = get_instance(model_arch, "arch", config)

        # setup model with device
        model, device = setup_device(model, config["target_device"])

        model_params = setup_model_params(model, config["optimizer"])
        optimizer = get_instance(optim, "optimizer", config, model_params)

        self.transforms = get_instance(model_transforms, "transforms", config)

        # train and test dataloaders
        self.data_loader = get_instance(
            model_data_loaders, "data_loader", config, self.transforms
        )

        train_loader, test_loader = self.data_loader.get_loaders()

        # Loss Function
        criterion = getattr(model_loss, config["criterion"])

        logger.info("Intializing the Trainer")
        self.trainer = Trainer(
            model, optimizer, criterion, config, device, train_loader, test_loader
        )

    def model_summary(self, input_size):
        summary(self.trainer.model, input_size)

    def plot_metrics(self):
        logger.info("Plotting the Metrics.")
        plt = plot.plot_metrics(self.trainer.train_metric, self.trainer.test_metric)
        return plt
