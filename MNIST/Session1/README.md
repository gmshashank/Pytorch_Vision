# Session 1 - MNIST 99.4% Test Accuracy

###	Objective:
For a Basic Network, achieve an accuracy of **99.4%** on the **MNIST** dataset with the following constraints:

- 99.4% validation accuracy
- Less than 20k Parameters
- Less than 20 Epochs
- No fully connected layers


###	Solution: 
Below are 10 Code Iterations to target this problem. 

###Iteration 1 - Setup

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmshashank/pytorch_vision/blob/main/MNIST/Session1/1_1_Setup.ipynb)

-   #### Target
	-   Get the set-up right
	-   Set Transforms
	-   Set Data Loader
	-   Set Basic Working Code
	-   Set Basic Training  & Test Loop

-   #### Results:
	-   Parameters: 6.3M (6,379,786)
	-   Best Training Accuracy: 100.00%
	-   Best Test Accuracy: 99.36%

-   #### Analysis:
	-   Extremely Heavy Model for such a problem
	-   Model is over-fitting


###Iteration 2 - Basic_Skeleton

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmshashank/pytorch_vision/blob/main/MNIST/Session1/1_2_Basic_Skeleton.ipynb)

-   #### Target
	-   Get the basic skeleton right. We will try and avoid changing this skeleton as much as possible. 

-   #### Results:
	-   Parameters: 194k (194,884)
	-   Best Training Accuracy: 99.60%
	-   Best Test Accuracy: 98.80%

-   #### Analysis:
	-   The model is still large, but working. 
	-   We see some over-fitting


###Iteration 3 - Lighter_Model
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmshashank/pytorch_vision/blob/main/MNIST/Session1/1_3_Lighter_Model.ipynb)

###Iteration 4 - BatchNorm
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmshashank/pytorch_vision/blob/main/MNIST/Session1/1_4_BatchNorm.ipynb)

###Iteration 5 - Regularization
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmshashank/pytorch_vision/blob/main/MNIST/Session1/1_5_Regularization.ipynb)

###Iteration 6 - Global_Average_Pooling
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmshashank/pytorch_vision/blob/main/MNIST/Session1/1_6_Global_Average_Pooling.ipynb)

###Iteration 7 - Increasing_Capacity
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmshashank/pytorch_vision/blob/main/MNIST/Session1/1_7_Increasing_Capacity.ipynb)

###Iteration 8 - Final_Model
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmshashank/pytorch_vision/blob/main/MNIST/Session1/1_8_Final_Model.ipynb)

###Iteration 9 - Image_Augmentation
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmshashank/pytorch_vision/blob/main/MNIST/Session1/1_9_Image_Augmentation.ipynb)

###Iteration 10 - LR_scheduler
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmshashank/pytorch_vision/blob/main/MNIST/Session1/1_10_LR_scheduler.ipynb)












