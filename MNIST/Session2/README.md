# Session 2 - MNIST 99.4% Test Accuracy

###	Objective:
For a Basic Network, achieve an accuracy of **99.4%** on the **MNIST** dataset with the following constraints:

- 99.4% validation accuracy
- Less than 10k Parameters
- Less than 15 Epochs
- No fully connected layers


###	Solution: 
Below are 4 Code Iterations to target this problem. 

### Iteration 1 - Basic Skeleton

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmshashank/pytorch_vision/blob/main/MNIST/Session2/1_BASIC_SKELETON.ipynb)

-   #### Target
	-   Get the set-up right
	-   Set Transforms
	-   Set Data Loader
	-   Set Basic Working Code
	-   Set Basic Training  & Test Loop
	-   Get the basic skeleton right. We will try and avoid changing this skeleton as much as possible 

-   #### Results:
	-   Parameters: 194k (194,884)
	-   Best Training Accuracy: 99.44%
	-   Best Test Accuracy: 98.80%

-   #### Analysis:
	-   The model is still large, but working 
	-   We see some over-fitting


### Iteration 2 - Lighter Model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmshashank/pytorch_vision/blob/main/MNIST/Session2/2_LIGHTER_MODEL.ipynb)

-   #### Target
	-   Making the model lighter 

-   #### Results:
	-   Parameters: 14k (13,752)
	-   Best Training Accuracy: 99.01%
	-   Best Test Accuracy: 98.72%

-   #### Analysis:
	-   The model is better. 
	-   No over-fitting, model is capable if pushed further


### Iteration 3 - Global Average Pooling (GAP)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmshashank/pytorch_vision/blob/main/MNIST/Session1/1_6_Global_Average_Pooling.ipynb)

-   #### Target
	-   Add GAP and remove the last BIG kernel

-   #### Results:
	-   Parameters: 10k (9,752)
	-   Best Training Accuracy: 98.166%
	-   Best Test Accuracy: 98.28%

-   #### Analysis:
	-   Since we have reduced model capacity, reduction in performance is expected

	
### Iteration 4 - Regularization and BatchNormalization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmshashank/pytorch_vision/blob/main/MNIST/Session1/1_4_BatchNorm.ipynb)

-   #### Target
	-   Add Batch-norm to increase model efficiency
	-   Add Regularization, Dropout = 5%

-   #### Results:
	-   Parameters: 10k (9,920)
	-   Best Training Accuracy: 99.27%
	-   Best Test Accuracy: 99.42%

-   #### Analysis:
	-   The model is working.
	-   Regularization is working.
	-   It is not over-fitting

