import torch

# from pathlib import Path
from typing import List, Tuple

from utils.logger import setup_logger
from utils.metrics import Metrics

logger = setup_logger(__name__)


class BaseTrainer:
    def __init__(self, model, optimizer, criterion, config, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = device
        self.epochs = config["training"]["epochs"]
        self.metrics = Metrics()
        self.lr_scheduler = None

    def train(self) -> Tuple[List, List]:
        logger.info("Starting the Training.")
        logger.info(f"Training the model for {self.epochs} epochs.")

        train_loss = []
        train_accuracy = []
        test_loss = []
        test_accuracy = []
        lr_metric = []

        for epoch in range(0, self.epochs):

            logger.info(f"Training Epoch: {epoch}.")

            if self.lr_scheduler:
                lr_value = [group["lr"] for group in self.optimizer.param_groups][0]
                logger.info(f"LR was set to: {lr_value}")
                lr_metric.append(lr_value)

            train_metric = self._train_epoch(epoch)

            logger.info(f"Test Epoch: {epoch}.")
            test_metric = self._test_epoch(epoch)

            if isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                val_loss = test_metric[2]
                self.lr_scheduler.step(val_loss)

            train_loss.extend(train_metric[0])
            train_accuracy.extend(train_metric[1])
            test_loss.extend(test_metric[0])
            test_accuracy.extend(test_metric[1])

        self.train_metric = (train_loss, train_accuracy)
        self.test_metric = (test_loss, test_accuracy)
        self.lr_metric = lr_metric

        # print("Training Done base.")
        return (self.train_metric, self.test_metric)

    def _train_epoch(self, epoch: int) -> dict:
        raise NotImplementedError

    def _test_epoch(self, epoch):
        raise NotImplementedError
