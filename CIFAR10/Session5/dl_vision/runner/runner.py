import math
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as model_scheduler
import pprint

import matplotlib.pyplot as plt

# import model.model as model_arch
import model.loss as model_loss
import data_loader.augmentations as augmentations
import data_loader.data_loaders as model_data_loaders
import utils.plot as plot

from trainer.trainer import Trainer
from torchsummary import summary

from utils.config import get_instance, setup_seed, setup_device, setup_model_params
from utils.logger import setup_logger
from utils.grad_cam import get_gradcam, plot_gradcam
from utils.lr_finder import LRFinder

logger = setup_logger(__name__)


class Runner:
    def __init__(self, config, custom_model=False):
        self.config = config
        self.custom_model = custom_model

    def setup_train(self):
        config = self.config
        logger.info("Training Configuration")

        if self.custom_model is True:
            import model.custom_model as model_arch
        else:
            import model.model as model_arch

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

        self.transforms = get_instance(augmentations, "transforms", config)

        # train and test dataloaders
        self.data_loader = get_instance(
            model_data_loaders, "data_loader", config, self.transforms
        )

        train_loader, test_loader = self.data_loader.get_loaders()

        # Loss Function
        criterion = getattr(model_loss, config["criterion"])

        batch_scheduler = False
        if config["lr_scheduler"]["type"] == "OneCycleLR":
            logger.info("Building: torch.optim.lr_scheduler.OneCycleLR")
            max_at_epoch = config["lr_scheduler"]["max_lr_at_epoch"]
            pct_start = (
                max_at_epoch / config["training"]["epochs"] if max_at_epoch else 0.8
            )
            scheduler_config = config["lr_scheduler"]["args"]
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=scheduler_config["max_lr"],
                steps_per_epoch=len(train_loader),
                pct_start=pct_start,
                epochs=config["training"]["epochs"],
            )
            batch_scheduler = True
        else:
            lr_scheduler = get_instance(
                model_scheduler, "lr_scheduler", config, optimizer
            )

        logger.info("Intializing the Trainer")
        self.trainer = Trainer(
            model,
            optimizer,
            criterion,
            config,
            device,
            train_loader,
            test_loader,
            lr_scheduler=lr_scheduler,
            batch_scheduler=batch_scheduler,
        )

    def model_summary(self, input_size):
        summary(self.trainer.model, input_size)

    def plot_metrics(self):
        logger.info("Plotting the Metrics.")
        plt = plot.model_metrics(self.trainer.train_metric, self.trainer.test_metric)
        return plt

    def plot_gradcam(self, target_layers):
        logger.info("Plotting GradCAM.")

        data, target = next(iter(self.trainer.test_loader))
        data, target = data.to(self.trainer.device), target.to(self.trainer.device)

        logger.info("Plotting for 5 Samples.")
        data = data[:5]
        target = target[:5]

        # get generated GradCAM data
        gcam_layers, predicted_probs, predicted_classes = get_gradcam(
            data, target, self.trainer.model, self.trainer.device, target_layers
        )

        # get the denomarlization function
        unorm = augmentations.UnNormalize(
            mean=self.transforms.mean, std=self.transforms.std
        )
        plot_gradcam(
            gcam_layers,
            data,
            target,
            predicted_classes,
            self.data_loader.class_names,
            unorm,
        )

    def plot_misclassified(self, target_layers):

        assert self.trainer.model is not None
        logger.info("Model Misclassified Images.")
        misclassified = []
        misclassified_target = []
        misclassified_predictions = []

        model, device = self.trainer.model, self.trainer.device

        model.eval()

        with torch.no_grad():
            for data, target in self.trainer.test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                predictions = output.argmax(dim=1, keepdim=True)
                misclassified_list = target.eq(predictions.view_as(target)) == False

                misclassified.append(data[misclassified_list])
                misclassified_target.append(target[misclassified_list])
                misclassified_predictions.append(predictions[misclassified_list])

        misclassified = torch.cat(misclassified)
        misclassified_target = torch.cat(misclassified_target)
        misclassified_predictions = torch.cat(misclassified_predictions)

        logger.info("Selecting 25 misclassified Samples.")

        data = misclassified[:25]
        target = misclassified_target[:25]

        # get generated GradCAM data
        gcam_layers, predicted_probs, predicted_classes = get_gradcam(
            data, target, self.trainer.model, self.trainer.device, target_layers
        )

        # get denormalization function
        unorm = augmentations.UnNormalize(
            mean=self.transforms.mean, std=self.transforms.std
        )

        plot_gradcam(
            gcam_layers,
            data,
            target,
            predicted_classes,
            self.data_loader.class_names,
            unorm,
        )

    def find_lr(self):
        logger.info("Finding best Learning Rate.")
        config = self.config

        if self.custom_model is True:
            import model.custom_model as model_arch

            print("custom_model")
        else:
            import model.model as model_arch

            print("local_model")

        # setup seed for reproducibility of results
        setup_seed(config["seed"])

        # create model instance
        model = get_instance(model_arch, "arch", config)

        # setup model with device
        model, device = setup_device(model, config["target_device"])

        model_params = setup_model_params(model, config["optimizer"])
        optimizer = get_instance(optim, "optimizer", config, model_params)

        self.transforms = get_instance(augmentations, "transforms", config)

        # Loss Function
        criterion = getattr(model_loss, config["criterion"])

        self.lr_finder = LRFinder(model, optimizer, criterion, device)

        lr_finder_epochs = config["lr_finder"]["epochs"]

        self.lr_finder.range_test(
            self.trainer.train_loader,
            start_lr=1e-3,
            end_lr=1,
            num_iter=len(self.trainer.test_loader) * lr_finder_epochs,
            step_mode="linear",
        )

        self.best_lr = self.lr_finder.history["lr"][
            self.lr_finder.history["loss"].index(self.lr_finder.best_loss)
        ]
        sorted_lrs = [
            x
            for _, x in sorted(
                zip(self.lr_finder.history["loss"], self.lr_finder.history["lr"])
            )
        ]

        logger.info(f"sorted lrs: {sorted_lrs[:10]}")
        logger.info(f"best lr: {self.best_lr}")
        logger.info("plotting lr_finder")

        self.lr_finder.plot()

        # reset the model and optimizer
        self.lr_finder.reset()
        plt.show()

        del model, optimizer, criterion

    def train_lr(self, use_best_lr=False, lr_value=None):

        if use_best_lr and self.best_lr is not None:
            logger.info(f"Using max_lr: {self.best_lr}")
            logger.info(f"Using min_lr: {self.best_lr/30}")
            logger.info(f"Using initial_lr: {self.best_lr/20}")
            for param_group in self.trainer.optimizer.param_groups:
                param_group["lr"] = self.best_lr / 10
                param_group["max_lr"] = self.best_lr
                param_group["min_lr"] = self.best_lr / 30
                param_group["initial_lr"] = self.best_lr / 20

        if not use_best_lr and lr_value is not None:
            for param_group in self.trainer.optimizer.param_groups:
                param_group["lr"] = lr_value

        self.trainer.train()
        logger.info("Finished.")
