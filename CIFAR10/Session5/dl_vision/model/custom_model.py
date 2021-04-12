from utils.logger import setup_logger
from base.base_model import BaseModel
import torch.nn as nn
import torch.nn.functional as F

from .resnet import *
from .resnet_custom import *

def CIFAR10_ResNet18():
    return ResNet18()


def CIFAR10_ResNet34():
    return ResNet34()


def CIFAR10_ResNet50():
    return ResNet50()

def CIFAR10_ResNet_custom():
    return ResNet_custom()
