# Session 1 - CIFAR10 85.0% Test Accuracy

###	Objective:
For a Basic Network, achieve an accuracy of **80.0%** on the **CIFAR10** dataset with the following constraints:

- Make the code modular
- change the architecture to B1-B2-B3-B4-Out (basically 3 MPs in between 4 blocks)
- total RF must be more than 44
- one of the layers must use Depthwise Separable Convolution
- one of the layers must use Dilated Convolution
- use GAP (compulsory):- add FC after GAP to target number of classes (optional)
- achieve 80% accuracy, as many epochs as you want. Total Params to be less than 1M. 


## Cifar 10 dataset 

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

Here are the classes in the dataset:
- airplane 										
- automobile 										
- bird 										
- cat 										
- deer 										
- dog 										
- frog 										
- horse 										
- ship 										
- truck

(ref - https://www.cs.toronto.edu/~kriz/cifar.html)
