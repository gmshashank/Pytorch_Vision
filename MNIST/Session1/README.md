# Session 1 - MNIST 99.4% Test Accuracy

###	Objective:
For a Basic Network, achieve an accuracy of **99.4%** on the **MNIST** dataset with the following constraints:

- 99.4% validation accuracy
- Less than 20k Parameters
- Less than 20 Epochs
- No fully connected layers


###	Solution: 
Below are 10 Code Iterations to target this problem. 

### Iteration 1 - Basic Setup

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


### Iteration 2 - Basic Skeleton

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmshashank/pytorch_vision/blob/main/MNIST/Session1/1_2_Basic_Skeleton.ipynb)

-   #### Target
	-   Get the basic skeleton right. We will try and avoid changing this skeleton as much as possible 

-   #### Results:
	-   Parameters: 194k (194,884)
	-   Best Training Accuracy: 99.60%
	-   Best Test Accuracy: 98.80%

-   #### Analysis:
	-   The model is still large, but working 
	-   We see some over-fitting


### Iteration 3 - Lighter Model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmshashank/pytorch_vision/blob/main/MNIST/Session1/1_3_Lighter_Model.ipynb)

-   #### Target
	-   Making the model lighter 

-   #### Results:
	-   Parameters: 11k (10,790)
	-   Best Training Accuracy: 99.01%
	-   Best Test Accuracy: 98.55%

-   #### Analysis:
	-   The model is better. 
	-   No over-fitting, model is capable if pushed further


### Iteration 4 - BatchNormalization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmshashank/pytorch_vision/blob/main/MNIST/Session1/1_4_BatchNorm.ipynb)

-   #### Target
	-   Add Batch-norm to increase model efficiency

-   #### Results:
	-   Parameters: 11k (10,970)
	-   Best Training Accuracy: 99.94%
	-   Best Test Accuracy: 98.34%

-   #### Analysis:
	-   We have started to see over-fitting now 
	-   Even if the model is pushed further, it won't be able to get to 99.4


### Iteration 5 - Regularization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmshashank/pytorch_vision/blob/main/MNIST/Session1/1_5_Regularization.ipynb)

-   #### Target
	-   Add Batch-norm to increase model efficiency

-   #### Results:
	-   Parameters: 11k (10,970)
	-   Best Training Accuracy: 99.44%
	-   Best Test Accuracy: 98.35%

-   #### Analysis:
	-   Regularization is working. But with current capacity, it is not possible to push it further
	-   We are also not using Global Average Pooling(GAP), but still using big sized kernel


### Iteration 6 - Global Average Pooling (GAP)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmshashank/pytorch_vision/blob/main/MNIST/Session1/1_6_Global_Average_Pooling.ipynb)

-   #### Target
	-   Add GAP and remove the last BIG kernel

-   #### Results:
	-   Parameters: 6k (6,070)
	-   Best Training Accuracy: 98.74%
	-   Best Test Accuracy: 98.54%

-   #### Analysis:
	-   Since we have reduced model capacity, reduction in performance is expected
	-   We are also not using Global Average Pooling(GAP), but still using big sized kernel


### Iteration 7 - Increasing Capacity

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmshashank/pytorch_vision/blob/main/MNIST/Session1/1_7_Increasing_Capacity.ipynb)

-   #### Target
	-    Increase model capacity by Adding more layers at the end

-   #### Results:
	-   Parameters: 12k (11,994)
	-   Best Training Accuracy: 99.27%
	-   Best Test Accuracy: 98.96%

-   #### Analysis:
	-   The model still showing over-fitting, possibly DropOut is not working as expected. We don't know which layer is causing over-fitting. 
		Adding it to a specific layer wasn't a great idea
	-   Quite Possibly we need to add more capacity, especially at the end
	-   Closer analysis of MNIST can also reveal that just at RF of 5x5 we start to see patterns forming
	-   We can also increase the capacity of the model by adding a layer after GAP!

	
### Iteration 8 - Final_Model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmshashank/pytorch_vision/blob/main/MNIST/Session1/1_8_Final_Model.ipynb)

-   #### Target
	-   Increase model capacity at the end (add layer after GAP)
	-   Perform MaxPooling at RF=5
	-   Fix DropOut, add it to each layer

-   #### Results:
	-   Parameters: 14k (13,808)
	-   Best Training Accuracy: 99.36%
	-   Best Test Accuracy: 99.49%

-   #### Analysis:
	-   The model is working.
	-   It is not over-fitting


### Iteration 9 - Image_Augmentation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmshashank/pytorch_vision/blob/main/MNIST/Session1/1_9_Image_Augmentation.ipynb)

-   #### Target
	-    Add rotation, our guess is that 5-7 degrees should be sufficient 

-   #### Results:
	-   Parameters: 14k (13,808)
	-   Best Training Accuracy: 99.14%
	-   Best Test Accuracy: 99.44%

-   #### Analysis:
	-   The model is under-fitting now. This is fine, as we know we have made our train data harder
	-   The test accuracy is also up, which means our test data had few images which had transformation difference w.r.t. train dataset


### Iteration 10 - LR_scheduler

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmshashank/pytorch_vision/blob/main/MNIST/Session1/1_10_LR_scheduler.ipynb)

-   #### Target
	-    Add LR Scheduler 

-   #### Results:
	-   Parameters: 14k (13,808)
	-   Best Training Accuracy: 99.16%
	-   Best Test Accuracy: 99.42%

-   #### Analysis:
	-   Finding a good LR schedule is hard. We have tried to make it effective by reducing LR by 10th after the 6th epoch. 
		It did help in getting to 99.4 or more faster, but final accuracy is not more than 99.5. 

	