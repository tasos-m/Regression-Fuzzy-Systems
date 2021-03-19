# Computational Intelligence - Regression Assignment
In this assignment, fuzzy TSK models are built to fit nonlinear, multivariable functions, in order to solve regression problem.

## Part 1
The first part of the assignment is about the training process and the evaluation of four models. 
>Dataset used: [Airfoil Self-Noise Dataset](https://archive.ics.uci.edu/ml/datasets/airfoil+self-noise) 

During the experiment these 4 models were tested:
* **TSK Model 1:** 2 Gaussian Input member functions and Singleton Output (constant)
* **TSK Model 2:** 3 Gaussian Input member functions and Singleton Output (constant)
* **TSK Model 3:** 2 Gaussian Input member functions and Polynomial Output (linear)
* **TSK Model 4:** 3 Gaussian Input member functions and Polynomial Output (linear)

### Metrics of the 4 models
|  | **RMSE** | **NMSE** | **NDEI** | **R<sup>2</sup>** |
| --- | --- | --- | --- | --- |
| **TSK Model 1**	| 3.8397 | 0.3379	| 0.5818	| 0.6621 |
| **TSK Model 2**	| 3.5005 | 0.2808	| 0.5299	| 0.7192 |
| **TSK Model 3** | 2.6932 | 0.1662	| 0.4077	| 0.8388 |
| **TSK Model 4** | 3.9961 | 0.366 |	0.605	| 0.634 |


## Part 2
In the second part, a large dataset of 11500 samples and 179 feautures is used. Thus it is important to reduce the number of feautures we use, to avoid the curse of dimensionality and the rule explosion. So before we train any model, we apply the _relieff_ function of the MATLAB Toolkit that ranks the importance of predictors, in order to choose the most important ones. Then, we devide the input space using Subtractive Clustering Technique, that is defined by the parameter "Range of influence of the cluster center". So we apply a Grid Search and 5-Fold Cross Validation to find which pair of range [0.3 0.5 0.8] and number of feautures [3 5 8 10] has the best performance, based on the validation error. Finally, we train the best model using the best pair of parameters accoridng to the Grid Search (range = 0.3 and number of fetures = 10), evaluate the performance and comment the results.   
>Dataset used: [Superconductivty Dataset](https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition) 

### Metrics of the final model

| **RMSE** | **NMSE** | **NDEI** | **R<sup>2</sup>** |
| --- | --- | --- | --- |
| 14.7679 | 0.1866	| 0.432	| 0.8134 |

### Split Scale
In both parts we use the split_scale.m function to split the data in training, validation and checking data and there is an option to normalize or standardize them.

