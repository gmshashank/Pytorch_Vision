# Session 1 - TinyImageNet 50.0% Test Accuracy using ResNet18 model, Data Augmentations and OneCyleLR

## TOC

1. [Overview](#overview)
1. [Model Summary](#Model-Summary)
1. [Loss Graphs](#loss-graphs)
1. [Samples](#samples)
1. [References](#references)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmshashank/Pytorch_Vision/blob/main/TinyImageNet/Session1/TinyImageNet_session1.ipynb)


## Overview

Train a Neural Network with the following constraints:

1. Use your data loader, model loading, train, and test code to train custom ResNet model on TinyImageNet dataset
2. Apply data Transformations using Albumentations library
3. Implement LR Finder, ReduceLROnPLateau and find best LR to train model
4. Use One Cycle Policy
5. Use SGD with Momentum.
6. To achieve 50% Test accuracy.
7. Plot images using Grad-CAM.

**Tiny-ImageNet Dataset**

Stanford prepared the Tiny ImageNet dataset for their CS231n course. 
The dataset spans 200 image classes with 500 training examples per class. 
The dataset also has 50 validation and 50 test examples per class.
The images are down-sampled to 64x64 pixels vs. 256x256 for full ImageNet. 
The full ImageNet dataset has 1000 classes vs. 200 classes in Tiny ImageNet.

**Deep Residual Learning for Image Recognition - (ResNet)**
 
Deeper neural networks are more difficult to train. One big problem of a deep network is the vanishing gradient problem.
To solve this problem, the authors proposed to use a reference to the previous layer to compute the output at a given layer. 
In ResNet, the output from the previous layer, called residual, is added to the output of the current layer.

**Cyclical Learning Rates for Training Neural Networks**

A PyTorch implementation of the learning rate range test detailed in [Cyclical Learning Rates for 
Training Neural Networks](https://arxiv.org/abs/1506.01186) by Leslie N. Smith and the tweaked version used by [fastai](https://github.com/fastai/fastai).
The learning rate range test is a test that provides valuable information about the optimal learning rate. 
During a pre-training run, the learning rate is increased linearly or exponentially between two boundaries. The low initial learning rate allows 
the network to start converging and as the learning rate is increased it will eventually be too large and the network will diverge.
Typically, a good static learning rate can be found half-way on the descending loss curve.

For cyclical learning rates (also detailed in Leslie Smith's paper) where the learning rate is cycled between two boundaries `(start_lr, end_lr)`,
the author advises the point at which the loss starts descending and the point at which the loss stops descending or becomes ragged for `start_lr` and `end_lr` respectively.

*** Leslie Smith's approach ***
Increases the learning rate linearly and computes the evaluation loss for each learning rate. lr_finder.plot() plots the evaluation loss versus learning rate. 
This approach typically produces more precise curves because the evaluation loss is more susceptible to divergence but it takes significantly longer to perform the test, 
especially if the evaluation dataset is large.

```python
from torch_lr_finder import LRFinder

model = ...
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-2)
lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(trainloader, val_loader=val_loader, end_lr=1, num_iter=100, step_mode="linear")
lr_finder.plot(log_lr=False)
lr_finder.reset()
```

*** Tweaked version from fastai ***
Increases the learning rate in an exponential manner and computes the training loss for each learning rate. lr_finder.plot() plots the training loss versus logarithmic learning rate.

```python
from torch_lr_finder import LRFinder

model = ...
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-2)
lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(trainloader, end_lr=100, num_iter=100)
lr_finder.plot() # to inspect the loss-learning rate graph
lr_finder.reset() # to reset the model and optimizer to their initial state
```

## Model Summary

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
            Conv2d-4          [-1, 128, 32, 32]          73,728
         MaxPool2d-5          [-1, 128, 16, 16]               0
       BatchNorm2d-6          [-1, 128, 16, 16]             256
            Conv2d-7          [-1, 128, 16, 16]         147,456
       BatchNorm2d-8          [-1, 128, 16, 16]             256
            Conv2d-9          [-1, 128, 16, 16]         147,456
      BatchNorm2d-10          [-1, 128, 16, 16]             256
         ResBlock-11          [-1, 128, 16, 16]               0
       LayerBlock-12          [-1, 128, 16, 16]               0
           Conv2d-13          [-1, 256, 16, 16]         294,912
        MaxPool2d-14            [-1, 256, 8, 8]               0
      BatchNorm2d-15            [-1, 256, 8, 8]             512
             ReLU-16            [-1, 256, 8, 8]               0
           Conv2d-17            [-1, 512, 8, 8]       1,179,648
        MaxPool2d-18            [-1, 512, 4, 4]               0
      BatchNorm2d-19            [-1, 512, 4, 4]           1,024
           Conv2d-20            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-21            [-1, 512, 4, 4]           1,024
           Conv2d-22            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-23            [-1, 512, 4, 4]           1,024
         ResBlock-24            [-1, 512, 4, 4]               0
       LayerBlock-25            [-1, 512, 4, 4]               0
        MaxPool2d-26            [-1, 512, 1, 1]               0
           Conv2d-27             [-1, 10, 1, 1]           5,120
================================================================
Total params: 6,573,120
Trainable params: 6,573,120
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 6.13
Params size (MB): 25.07
Estimated Total Size (MB): 31.22
----------------------------------------------------------------
```

## Loss Graphs

![Loss Graphs](https://github.com/gmshashank/Pytorch_Vision/blob/main/TinyImageNet/Session1/images/metrics.png)

## Samples

![LR](https://github.com/gmshashank/Pytorch_Vision/blob/main/TinyImageNet/Session1/images/LR.png)


![OneCycleLR](https://github.com/gmshashank/Pytorch_Vision/blob/main/TinyImageNet/Session1/images/OneCycleLR.png)

## References

- [Tiny-ImageNet Dataset](http://cs231n.stanford.edu/tiny-imagenet-200.zip)

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [ResNet models](https://github.com/kuangliu/pytorch-cifar)

- [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)
- [PyTorch learning rate finder](https://github.com/davidtvs/pytorch-lr-finder)

- [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) 
- [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay](https://arxiv.org/abs/1803.09820)