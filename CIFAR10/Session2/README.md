# Session 2 - CIFAR10 85.0% Test Accuracy using ResNet18

## TOC

1. [Overview](#overview)
1. [Model Summary](#Model-Summary)
1. [Loss Graphs](#loss-graphs)
1. [Samples](#samples)
1. [References](#references)

## Overview

Train a Neural Network with the following constraints:

1. Use your data loader, model loading, train, and test code to train ResNet18 on Cifar10
2. The Target is to achieve 85% accuracy.

**Deep Residual Learning for Image Recognition - (ResNet)**
 
Deeper neural networks are more difficult to train. One big problem of a deep network is the vanishing gradient problem.
To solve this problem, the authors proposed to use a reference to the previous layer to compute the output at a given layer. 
In ResNet, the output from the previous layer, called residual, is added to the output of the current layer.


## Model Summary

		----------------------------------------------------------------
				Layer (type)               Output Shape         Param #
		================================================================
					Conv2d-1           [-1, 64, 32, 32]           1,728
			   BatchNorm2d-2           [-1, 64, 32, 32]             128
					Conv2d-3           [-1, 64, 32, 32]          36,864
			   BatchNorm2d-4           [-1, 64, 32, 32]             128
					Conv2d-5           [-1, 64, 32, 32]          36,864
			   BatchNorm2d-6           [-1, 64, 32, 32]             128
				BasicBlock-7           [-1, 64, 32, 32]               0
					Conv2d-8           [-1, 64, 32, 32]          36,864
			   BatchNorm2d-9           [-1, 64, 32, 32]             128
				   Conv2d-10           [-1, 64, 32, 32]          36,864
			  BatchNorm2d-11           [-1, 64, 32, 32]             128
			   BasicBlock-12           [-1, 64, 32, 32]               0
				   Conv2d-13          [-1, 128, 16, 16]          73,728
			  BatchNorm2d-14          [-1, 128, 16, 16]             256
				   Conv2d-15          [-1, 128, 16, 16]         147,456
			  BatchNorm2d-16          [-1, 128, 16, 16]             256
				   Conv2d-17          [-1, 128, 16, 16]           8,192
			  BatchNorm2d-18          [-1, 128, 16, 16]             256
			   BasicBlock-19          [-1, 128, 16, 16]               0
				   Conv2d-20          [-1, 128, 16, 16]         147,456
			  BatchNorm2d-21          [-1, 128, 16, 16]             256
				   Conv2d-22          [-1, 128, 16, 16]         147,456
			  BatchNorm2d-23          [-1, 128, 16, 16]             256
			   BasicBlock-24          [-1, 128, 16, 16]               0
				   Conv2d-25            [-1, 256, 8, 8]         294,912
			  BatchNorm2d-26            [-1, 256, 8, 8]             512
				   Conv2d-27            [-1, 256, 8, 8]         589,824
			  BatchNorm2d-28            [-1, 256, 8, 8]             512
				   Conv2d-29            [-1, 256, 8, 8]          32,768
			  BatchNorm2d-30            [-1, 256, 8, 8]             512
			   BasicBlock-31            [-1, 256, 8, 8]               0
				   Conv2d-32            [-1, 256, 8, 8]         589,824
			  BatchNorm2d-33            [-1, 256, 8, 8]             512
				   Conv2d-34            [-1, 256, 8, 8]         589,824
			  BatchNorm2d-35            [-1, 256, 8, 8]             512
			   BasicBlock-36            [-1, 256, 8, 8]               0
				   Conv2d-37            [-1, 512, 4, 4]       1,179,648
			  BatchNorm2d-38            [-1, 512, 4, 4]           1,024
				   Conv2d-39            [-1, 512, 4, 4]       2,359,296
			  BatchNorm2d-40            [-1, 512, 4, 4]           1,024
				   Conv2d-41            [-1, 512, 4, 4]         131,072
			  BatchNorm2d-42            [-1, 512, 4, 4]           1,024
			   BasicBlock-43            [-1, 512, 4, 4]               0
				   Conv2d-44            [-1, 512, 4, 4]       2,359,296
			  BatchNorm2d-45            [-1, 512, 4, 4]           1,024
				   Conv2d-46            [-1, 512, 4, 4]       2,359,296
			  BatchNorm2d-47            [-1, 512, 4, 4]           1,024
			   BasicBlock-48            [-1, 512, 4, 4]               0
				   Linear-49                   [-1, 10]           5,130
		================================================================
		Total params: 11,173,962
		Trainable params: 11,173,962
		Non-trainable params: 0
		----------------------------------------------------------------
		Input size (MB): 0.01
		Forward/backward pass size (MB): 11.25
		Params size (MB): 42.63
		Estimated Total Size (MB): 53.89
		----------------------------------------------------------------


## Loss Graphs

![Loss Graphs](https://github.com/gmshashank/Pytorch_Vision/blob/main/CIFAR10/Session2/images/metrics.png)

## Samples


## References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [ResNet models](https://github.com/kuangliu/pytorch-cifar)
