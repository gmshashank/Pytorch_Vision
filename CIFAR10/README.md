# [Session 1 - CIFAR10 85.0% Test Accuracy](Session1/README.md)

###	Objective:
For a Basic Network, achieve an accuracy of **80.0%** on the **CIFAR10** dataset with the following constraints:

1. Make the code modular
2. change the architecture to B1-B2-B3-B4-Out (basically 3 MPs in between 4 blocks)
3. total RF must be more than 44
4. one of the layers must use Depthwise Separable Convolution
5. one of the layers must use Dilated Convolution
6. use GAP :- add FC after GAP to target number of classes (optional)
7. achieve 80% accuracy, as many epochs as you want. Total Params to be less than 1M.

# [Session 2 - Test Accuracy of **85.0%** on the **CIFAR10** dataset using ResNet18](Session2/README.md)

###	Objective:
Train a Neural Network with the following constraints:

1. Use your data loader, model loading, train, and test code to train ResNet18 on Cifar10
2. The Target is to achieve 85% accuracy.


# [Session 3 - Test Accuracy of **85.0%** on the **CIFAR10** dataset using ResNet18 using **Augmentations**](Session3/README.md)

###	Objective:
Train a Neural Network with the following constraints:

1. Use your data loader, model loading, train, and test code to train ResNet18 on Cifar10
2. Apply data Transformations using Albumentations library
3. To achieve 85% Test accuracy.
4. Plot images using Grad-CAM.
