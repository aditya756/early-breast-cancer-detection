# early-breast-cancer-detection
## WORKFLOW:
Data gathering and pre-processing: Breast Cancer Wisconsin dataset is available on both Kaggle and UCI Machine Learning Repository, and is obtained from the same. The Wisconsin dataset is public, and various studies have already been conducted using the dataset. The dataset was converted into CSV format in order to examine the data in the Jupyter notebook. The dataset is required to be pre-processed in order to eliminate the id column, the last column with a null value, and any rows with some null value. In the end, 32 characteristics were employed in the detecting procedure. 

![](https://github.com/aditya756/early-breast-cancer-detection/blob/main/images/image2.png)

Figure 1. Flowchart showing basic methodology


## Classification Process: 
In the classification, five methods were used: Random Forest Classifier, AdaBoost,  XGBoost, Support Vector Machine, and Sequential Minimal Optimization. To achieve ideal accuracy, the algorithms were tuned at various hyper-parameters. Modifications were made to the model in order for the classifier-generated model to properly categorize the random data.

![](https://github.com/aditya756/early-breast-cancer-detection/blob/main/images/image3.png)

Figure 2. Methodology illustrating ML Classifiers

## Correlation:
To have a better understanding of the data, we created a correlation table between all of the elements and visualized it with pair plotting and a heatmap. Since the size of the correlation table and plot was huge we have shown a part of their result.


![](https://github.com/aditya756/early-breast-cancer-detection/blob/main/images/image4.png)

Figure. Correlation depiction between different features using pairplot


![](https://github.com/aditya756/early-breast-cancer-detection/blob/main/images/image5.png)

![](https://github.com/aditya756/early-breast-cancer-detection/blob/main/images/image6.png)

Fig. Heatmap of correlation between features


## TECHNIQUES / ALGORITHMS:
Below we have mentioned the basic working of the algorithms which are being implemented. Along with that, the hyperparameters of each algorithm are mentioned. Hyperparameters are the specific parameters which increase the performance of the respective algorithms.
Random Forest Classifier:
Working:
Selection of random samples from the dataset having n number of records.
Construction of individual decision trees for each record.
Output generated from each individual decision tree.
Majority Voting or Averaging is being done to select the final result.
	Important Hyperparameters involved: n_estimators, max_features, min_samples_leaf
	
### AdaBoost:
Working:
Equal weights are assigned to each record in the dataset.
On the subset of data, a model is trained. Predictions are made on the whole dataset using the same model.
Comparison of prediction and actual values to calculate the errors.
Higher weights assigned to the records which were predicted incorrectly, while constructing the next model.
In general, higher the error, more the weight assigned to that record, similarly lower the error, less weight is assigned.
Process repeated until the number of estimators doesn’t reach the maximum limit or error function doesn’t change.
Important Hyperparameters involved: n_estimators, random_state

### XGBoost:
	Working: 
Predictions from many models are merged into a single forecast.
Then iteratively model each prediction depending on the mistake of its predecessor.
The predictors that perform better are given more weights.
A gradient descent approach is used in XGBoost to minimize the loss function.
Decision trees as weak predictors are being used.
	
Important Hyperparameters involved: booster, reg_alpha and reg_lambda, max_depth, subsample, num_estimators

### Support Vector Machine:
	Working: 
Support Vectors are used.
Say in a dataset we have two classes, red and black. We want to classify the new datapoint as either red or black. We try to find decision boundaries for the same.
The best hyperplane is the one that has the greatest distance between both classes.
We find different hyperplanes which classify the labels in the best way.
Select the one which has a maximum margin or is farthest from the data points.

Important Hyperparameters involved: c, gamma, kernel
