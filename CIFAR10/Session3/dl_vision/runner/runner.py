import math
import torch
import torch.optim as optim
import pprint

# import model.model as model_arch
import model.loss as model_loss
import data_loader.transforms as model_transforms
import data_loader.data_loaders as model_data_loaders
import utils.plot as plot

from trainer.trainer import Trainer
from torchsummary import summary

from utils.config import get_instance, setup_seed, setup_device, setup_model_params
from utils.logger import setup_logger
from utils.grad_cam import get_gradcam, plot_gradcam

logger = setup_logger(__name__)


class Runner:
    def __init__(self, config):
        self.config = config

    def setup_train(self, custom_model=False):
        config = self.config
        logger.info("Training Configuration")

        if custom_model:
            import model.custom_model as model_arch

            print("custom_model")
        else:
            import model.model as model_arch

            print("local_model")

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

        unorm = model_transforms.UnNormalize(
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
