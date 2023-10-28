# **PreLect : A Robust Approach for Feature Engineering in Sparse Matrices**


[![License](https://img.shields.io/badge/License-MIT-blue)](#license)


Table of Contents
====================
- [Installation](#installation)
- [General usage](#general-usage)
- [Lambda tuning](#lambda-tuning)
- [Selection profile](#selection-profile)
- [Citing Resources](#citing-resources)
- [Authors](#authors)
- [Found a Bug](#found-a-bug)

## Installation

  [Pytorch](https://pytorch.org/) is essential, please install it first.

  Install PreLect via package clone from GitHub repository.

```Shell
$ git clone https://github.com/YinchengChen23/PreLect
$ cd PreLect
$ pip install . 
```

## General usage
PreLect is a classifier based on logistic regression with an *11*-like penalty, designed with a scikit-learn-like API. We can set the parameters using the `PreLect` class and use the built-in function `.fit()` to fit the model to the training data. Eventually, the selection results are presented in the `.feature_set` and `.feature_sort` attributes.

```python
class PreLect(lmbd=1e-5, max_iter=10000, tol=1e-4, lr=0.001, alpha=0.9, epsilon=1e-8,
              device='cpu', echo=False)
```

| Argument | Type    | Default  | Description                                                      |
|----------|---------|----------|------------------------------------------------------------------|
| lmbd     | Float   | 1e-5     | Lambda, regularization intensity.                                |
| max_iter | Integer | 10000	  | Maximum number of iterations taken for the solvers to converge.  |
| tol      | Float   | 1e-4     | Tolerance for stopping criteria.                                 |
| lr       | Float 	 | 0.001    | Learning  rate in RMSprop optimizer.                             |
| alpha    | Float 	 | 0.9      | Smoothing constant in RMSprop optimizer.                         |
| epsilon  | Float   | 1e-8     | Term added to the denominator to improve numerical stability.    |
| device   | String  | cpu      | `cpu` is run with CPU, and `cuda` is run via GPU <br>  It is more effective to use GPU when the sample is more than 200 and the feature is more than 30,000. |
| echo     | Bool 	 | Fasle    | print out the final training result.                             |


| Attributes    | Type                              | Description                                  |
|---------------|-----------------------------------|----------------------------------------------|
| classes_      | dict : {0 : lable_0, 1 : label_1} | A dictionary of class labels corresponding to binary prediction. |
| n_iter_       | ndarray of shape (1, )            | The iteration number after optimization.     |
| loss_         | ndarray of shape (1, )            | The loss value after optimization.           |
| convergence_  | ndarray of shape (1, )            | The degree of convergence after optimization.|
| w             | ndarray of shape (n_features, )   | The weights of the features in the decision function. |
| b             | ndarray of shape (1, )            | The bias in the decision function.           |
| feature_set   | ndarray of shape (n_features, )   | The list to indicate the selected features, 1 is chosen and 0 is not. |
| feature_sort  | ndarray of shape (n_features, )   | The list of sorted features indices by importance. |

#### Examples
```python
>>> import os
>>> import time
>>> import scipy
>>> import numpy as np
>>> import pandas as pd
>>> import PreLect
>>> from PreLect.PL_utils import *

>>> url_1 = 'https://raw.githubusercontent.com/YinchengChen23/PreLect/main/data/crc_zeller/ASV_vst.txt'
>>> url_2 = 'https://raw.githubusercontent.com/YinchengChen23/PreLect/main/data/crc_zeller/ASV_table.txt'
>>> url_3 = 'https://raw.githubusercontent.com/YinchengChen23/PreLect/main/data/crc_zeller/metadata.txt'
>>> Data = pd.read_csv(url_1, sep = "\t"); Data = Data.T           # Variance-stabilizing transformation was conducted by DESeq2
>>> Data_std = scipy.stats.zscore(Data, axis=0, ddof=0)            # we using z-normalization data as input-data
>>> RawData = pd.read_csv(url_2, sep = "\t"); RawData = RawData.T  # Raw count data, was used as an assessment of prevalence
>>> Cohort = pd.read_csv(url_3, sep = "\t")                        # Metadata
>>> Label = Cohort['Class'].tolist()

>>> print('This data contains', Data_std.shape[0], 'samples and', Data_std.shape[1], 'features')
This data contains 129 samples and 6661 features

>>> print(Label[0:10], np.unique(Label))
['Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal'] ['Cancer' 'Normal']


>>> pvl = get_prevalence(RawData, np.arange(RawData.shape[0]))
>>> res = PreLect(lmbd = 1e-6, echo= True)
>>> start = time.time()
>>> res.fit(Data_std, Label, pvl)
minimum epoch =  9999 ; minimum lost =  6.27363842795603e-05 ; diff weight =  0.002454951871186495
>>> end = time.time()
>>> print('median of selected prevalence :',np.median([pvl[i]  for i, w in enumerate(res.feature_set) if w != 0]))
median of prevalence : 0.26356589147286824

>>> print('total selected feature :',np.sum(res.feature_set))
total selected feature : 605

>>> print("Total cost：%f sec" % (end - start))
Total cost：2.332477 sec

# importance ranking
>>> for ix in res.feature_sort:
        print(Data_std.columns[ix], res.w[ix])
ASV00013 1.0226521
ASV00003 0.8313215
ASV00043 0.81473553
ASV00054 0.7314421
ASV00017 0.719262
ASV00014 -0.65857583
ASV00144 0.6585081
.
.
.

#Export selection result
>>> res.writeList('/home/yincheng23/PreLect/data/crc_zeller/selectedList.txt', Data_std.columns)
# check here https://github.com/YinchengChen23/PreLect/blob/main/data/crc_zeller/selectedList.txt




# We also can do the split training and testing with PreLect
>>> from sklearn.model_selection import train_test_split
>>> train_X, test_X, train_y, test_y = train_test_split(Data_std, Label, test_size=0.2)
>>> train_ix = np.array([ix for ix, sample in enumerate(Data_std.index) if sample in train_X.index])
>>> pvl = get_prevalence(RawData, train_ix)
>>> res = PreLect(lmbd = 1e-6, echo= True)
>>> res.fit(train_X, train_y, pvl)
>>> print('median of prevalence :',np.median([pvl[i]  for i, w in enumerate(res.feature_set) if w != 0]))
median of prevalence : 0.2815533980582524

>>> print('total selected feature :',np.sum(res.feature_set))
total selected feature : 568

>>> metrics_dict = res.score(test_X, test_y)
>>> print(metrics_dict)
{'AUC': 0.6616541353383458,
 'AUPR': 0.7919543846915067,
 'MCC': 0.1822619328515436,
 'Precision': array([0.82608696, 0.81818182, 0.80952381, 0.85      , 0.84210526,
        0.83333333, 0.82352941, 0.8125    , 0.8       , 0.78571429,
        0.76923077, 0.75      , 0.72727273, 0.7       , 0.77777778,
        0.75      , 0.85714286, 0.83333333, 0.8       , 0.75      ,
        0.66666667, 0.5       , 1.        , 1.        ]),
 'Recall': array([1.        , 0.94736842, 0.89473684, 0.89473684, 0.84210526,
        0.78947368, 0.73684211, 0.68421053, 0.63157895, 0.57894737,
        0.52631579, 0.47368421, 0.42105263, 0.36842105, 0.36842105,
        0.31578947, 0.31578947, 0.26315789, 0.21052632, 0.15789474,
        0.10526316, 0.05263158, 0.05263158, 0.        ])}

>>> fig, ax = plt.subplots()
>>> ax.plot(metrics_dict['Recall'], metrics_dict['Precision'], color='purple')
>>> ax.set_title('Precision-Recall Curve')
>>> ax.set_ylabel('Precision')
>>> ax.set_xlabel('Recall')
>>> plt.show()
# You may notice that the accuracy of PreLect is not very high as
# it is a feature selector rather than a classifier.
```
<br>

> **WARNING**
> Since PreLect has not yet been established as a full scikit-learn API, so in the process of using it, you must first declare a class before doing the fit, you cannot do the fit directly.


```python
# Wrong way
>>> res = PreLect(lmbd=1e-5).fit(train_X, train_y, pvl_vec) 
# Correct way
>>> res = PreLect(lmbd=1e-5)
>>> res.fit(train_X, train_y, pvl_vec)
```
#### Methods

`fit(X, y, prevalence )  `

Fit the model according to the given training data.

|                 |                                                                    |
|-----------------|--------------------------------------------------------------------|
| **Parameters:** | **X : *{pd-dataframe, np-array or sparse matrix} of shape (n_samples, n_features)*** <br>  Training vector, where `n_samples` is the number of samples and `n_features` is the number of features. <br> <br> **y : *{list or np-array} of shape (n_samples, )*** <br> Target vector relative to X. <br> <br> **prevalence : *array of shape (n_features, )*** <br> The prevalence vector relative to each feature in X, which can obtain from `get_prevalence` function in this package. |
| **Returns:**    | **self** <br> Fitted estimator.                                    |

<br>

`get_y_array(label_list)`

Get the corresponding binary array in this model.

|                 |                                                                    |
|-----------------|--------------------------------------------------------------------|
| **Parameters:** | **label_list : *{list or np-array} of shape (n_samples, )***       |
| **Returns:**    | **y : *array of shape (n_samples, )***  <br> Converting a label string to a binary vector.  | 

<br>

`predict_proba(X)`

Probability estimates. The returned estimates for class representing 1.

|                 |                                                                    |
|-----------------|--------------------------------------------------------------------|
| **Parameters:** | **X : *{pd-dataframe, np-array or sparse matrix} of shape (n_samples, n_features)*** <br> The data matrix for which we want to get the predictions.                                                           |
| **Returns:**    | **proba : *array of shape (n_samples, )***  <br> Returns the probability of the sample for class 1. You can check which category represents as 1 from `.classes_` attributes in ADlasso class. | 

<br>

`predict(X)`

Predict class labels for samples in X.

|                 |                                                                    |
|-----------------|--------------------------------------------------------------------|
| **Parameters:** | **X : *{pd-dataframe, np-array or sparse matrix} of shape (n_samples, n_features)*** <br>  The data matrix for which we want to get the predictions.                                                           |
| **Returns:**    | **pred : *array of shape (n_samples, )***  <br> Vector containing the class labels for each sample. | 

<br>

`score(X, y)`

Return the metrics on the given test data and labels.

|                 |                                                                    |
|-----------------|--------------------------------------------------------------------|
| **Parameters:** | **X : *{pd-dataframe, np-array or sparse matrix} of shape (n_samples, n_features)*** <br> Test samples. <br><br> **y : *{list or np-arry} of shape (n_samples, )***  <br> True labels for X. |
| **Returns:**    | **score : *dict*** <br> A dictionary of measurements **{AUC, AUPR, MCC, precision, recall}** | 

<br>

`writeList(outpath, featureNameList)`



Automatically generates a list of feature selections.

|                 |                                                                    |
|-----------------|--------------------------------------------------------------------|
| **Parameters:** | **outpath : *String*** <br> File name and absolute path of the form to be exported. <br><br> **featureNameList (optional) : *{list or pd.columns} of shape (n_features, )***  <br> Feature names for data. |
| **Returns:**    | **selection list : *File*** <br> First column : Name or index of selected feature. <br> Second column : Weight of each feauture. <br> Third column : Tendency of each feature. | 

## Lambda tuning

Unlike the common strategies for parameters tuning (based on performance), Here, we propose a method to determine the parameter according to the variation of loss value. We observe that the number of selected features gradually decreases as lambda intensity from weak to strong, as shown below.
![alt](https://github.com/YinchengChen23/PreLect/blob/main/img/img1.png?raw=true)
So we design a function `AutoScanning` which can automatically scan the lambda from $10^{-10}$ to $1$, and identify the upper and lower boundaries representing lasso start filtering and lasso drop all features respectively (black dotted line). And divide `k` parts whitin this region as the examining lambdas.

`AutoScanning(X_input, X_raw, Y, step=50, device='cpu', training_echo=False, max_iter=10000, tol=1e-4, lr=0.001, alpha=0.9, epsilon=1e-8)`

|                 |                                                                    |
|-----------------|--------------------------------------------------------------------|
| **Parameters:** | **X_input : *{pd-dataframe, np-array or sparse matrix} of shape (n_samples, n_features)*** <br>  The normalized or transformed data for providing feature space.  <br> <br> **X_raw : *{pd-dataframe, np-array or sparse matrix} of shape (n_samples, n_features)*** <br>   The original, count data for calculating prevalence in each feature. <br> <br> **Y : *{list or np-array} of shape (n_samples, )*** <br> Target vector relative to X_input. <br> <br> **step : *Integer*** <br> split into k parts within upper and lower bounds as examining lambda.  <br> <br> **other prarameters : the basic seting in `PreLect` class**  |
| **Returns:**    | **log_lmbd_range : *np-array of shape (n_steps,)*** <br> The vector of examining lambda. |

Afterwards, we performed a k-fold cross validation (CV) for examining each lambda via function `LambdaTuning`, the metrics which including area under curve of Precision Recall Curve (PRC), Matthews Correlation Coefficient (MCC), minimal error and minimal Binary Cross Entropy loss (BCE) were evaluate in each examining lambda. Since this procedure is time consuming, we suggest running it with `nohup`, and we provide a `outdir` option to save the results to the folder you specify.

`LambdaTuning(X_input, X_raw, Y, lmbdrange, k_fold, outdir, device='cpu', training_echo=False, max_iter=10000, tol=1e-4, lr=0.001, alpha=0.9, epsilon=1e-8)` 

|                 |                                                                    |
|-----------------|--------------------------------------------------------------------|
| **Parameters:** | **X_input : *{pd-dataframe, np-array or sparse matrix} of shape (n_samples, n_features)*** <br>  The normalized or transformed data for providing feature space.  <br> <br> **X_raw : *{pd-dataframe, np-array or sparse matrix} of shape (n_samples, n_features)*** <br>   The original, count data for calculating prevalence in each feature. <br> <br> **Y : *{list or np-array} of shape (n_samples, )*** <br> Target vector relative to X_input. <br> <br> **lmbdrange : *np-array of shape (n_steps,)*** <br>  examining lambda vector. <br> <br> **k_fold : *Integer*** <br> split data into k parts for cross validation. <br> <br>  **outdir : *String*** <br>  Absolute-path of output folder  <br> <br> **other prarameters : the basic seting in `PreLect` class**  |
| **Returns:**    | **metrics_dict : *Dict*** <br>  {'Percentage', 'Prevalence', 'Feature_number', 'AUC', 'AUPR', 'MCC', 'loss_history', 'error_history', 'pairwiseMCC'}. |

As mentioned above, we also provide a function `get_tuning_result` to get the result from output folder, if you conduct `LambdaTuningViz` via `nohup`.

`get_tuning_result(result_path)`

|                 |                                                                      |
|-----------------|----------------------------------------------------------------------|
| **Parameters:** | **result_path : *String*** <br> Absolute pathoutput of result folder.|
| **Returns:**    | **metrics_dict : *Dict*** <br>  {'Percentage', 'Prevalence', 'Feature_number', 'AUC', 'AUPR', 'MCC', 'loss_history', 'error_history', 'pairwiseMCC'}. |

#### Examples
```python
>>> log_lmbd_range = AutoScanning(Data_std, RawData, Label, step=50)

>>> lmbd_range = np.exp(log_lmbd_range)
>>> print(lmbd_range)
array([1.00000000e-08, 1.26485522e-08, 1.59985872e-08, 2.02358965e-08,
       2.55954792e-08, 3.23745754e-08, 4.09491506e-08, 5.17947468e-08,
       6.55128557e-08, 8.28642773e-08, 1.04811313e-07, 1.32571137e-07,
       1.67683294e-07, 2.12095089e-07, 2.68269580e-07, 3.39322177e-07,
       4.29193426e-07, 5.42867544e-07, 6.86648845e-07, 8.68511374e-07,
       1.09854114e-06, 1.38949549e-06, 1.75751062e-06, 2.22299648e-06,
       2.81176870e-06, 3.55648031e-06, 4.49843267e-06, 5.68986603e-06,
       7.19685673e-06, 9.10298178e-06, 1.15139540e-05, 1.45634848e-05,
       1.84206997e-05, 2.32995181e-05, 2.94705170e-05, 3.72759372e-05,
       4.71486636e-05, 5.96362332e-05, 7.54312006e-05, 9.54095476e-05,
       1.20679264e-04, 1.52641797e-04, 1.93069773e-04, 2.44205309e-04,
       3.08884360e-04, 3.90693994e-04, 4.94171336e-04, 6.25055193e-04,
       7.90604321e-04, 1.00000000e-03])

>>> k_fold = 5
>>> outPath_dataSet = '/home/yincheng23/PreLect/data/crc_zeller/LmbdTuning'
>>> result_dict =LambdaTuning(Data_std, RawData, Label, lmbd_range, k_fold, outPath_dataSet)
# check here https://github.com/YinchengChen23/PreLect/blob/main/data/crc_zeller/LmbdTuning

>>> result_dict_recell = get_tuning_result(outPath_dataSet)
```

We provided user-friendly visualization function `LambdaTuningViz` to assist you for viewing the results, 9 measurements were tested in each lambda, a detailed description is provided below.

- Percentage : Percentage of features were selected in each lambda.
- Prevalence : The medium value of selected features prevalence in each lambda.
- Feature_number : The number of selected features in each lambda.
- loss_history : The minimum loss value of PreLect after training.
- error_history : The minimum iteration difference after training.
- pairwiseMCC : Compare Matthews correlation coefficient of selected feature within k-fold CV.

We also retrain a logistic classifier with the selected features, and test it with `AUC`, `AUPR` and `MCC`. mean and standard deviation in k-fold CV.

`LambdaTuningViz(result_dict, metric, savepath=None, fig_width=8, fig_height=4)`

|                 |                                                                      |
|-----------------|----------------------------------------------------------------------|
| **Parameters:** | **result_dict : *Dict*** <br> The output of `LambdaTuning` or `get_tuning_result` function which including 9 measurements for each examining lambda. <br><br> **metric : *String*** <br>{`'Percentage'`, `'Prevalence'`, `'Feature_number'`, `'AUC'`, `'AUPR'`, `'MCC'`, `'loss_history'`, `'error_history'`, `'pairwiseMCC'`} <br> chose one you want to check. <br><br> **savepath (optional) : *String, absolute-path*** <br> you can assign a path to save the figure  <br><br> |
| **Returns:**    | **Fig : *matplotlib.figure object***                                 |

#### Examples

`Fig = LambdaTuningViz(result_dict_recell, 'Feature_number')`
![alt](https://github.com/YinchengChen23/PreLect/blob/main/img/img2.png?raw=true)

`Fig = LambdaTuningViz(result_dict_recell, 'AUC')`
![alt](https://github.com/YinchengChen23/PreLect/blob/main/img/img3.png?raw=true)

`Fig = LambdaTuningViz(result_dict_recell, 'loss_history')`
![alt](https://github.com/YinchengChen23/PreLect/blob/main/img/img4.png?raw=true)

`Fig = LambdaTuningViz(result_dict_recell, 'error_history')`
![alt](https://github.com/YinchengChen23/PreLect/blob/main/img/img5.png?raw=true)

```python
# You can save the figure via
>>> outputPath = '/home/yincheng23/PreLect/img/img2.png'
>>> Fig = LambdaTuningViz(result_dict_recell, 'Feature_number', outputPath)

#or
>>> Fig = LambdaTuningViz(result_dict_recell, 'Feature_number')
>>> Fig.savefig(outputPath)
```

We recommend determining the optimal lambda value based on the inflection point of the BCE loss curve. This point marks the crucial balance where the regularization term outweighs the loss term. So we designed a function `lambda_decision` to find the inflection point by slicing the BCE loss curve into `k` part via segemented regression.

`lambda_decision(result_dict, k, savepath=None, fig_width=8, fig_height=4)`

|                 |                                                                      |
|-----------------|----------------------------------------------------------------------|
| **Parameters:** | **result_dict : *Dict*** <br> The output of `LambdaTuning` or `get_tuning_result` function which including 9 measurements for each examining lambda. <br><br> **k : *Integer*** <br> Decide n_segments to divide. <br><br> **savepath (optional) : *String, absolute-path*** <br> you can assign a path to save the figure  <br><br> |
| **Returns:**    |  **selected_lambda : *Float*** <br> Final lambda. <br><br> **Fig : *matplotlib.figure object*** |

```python
>>> outputPath = '/home/yincheng23/PreLect/img/img2.png'
>>> opt_lmbd, fig = lambda_decision(result_dict_recell, 5)
>>> print(opt_lmbd)
1.4563484775012424e-05

>>> pvl = get_prevalence(RawData, np.arange(RawData.shape[0]))
>>> opt_res = PreLect(lmbd = opt_lmbd, echo= True)
>>> start = time.time()
>>> opt_res.fit(Data_std, Label, pvl)
minimum epoch =  9999 ; minimum lost =  0.00022168313444126397 ; diff weight =  0.002862341469153762
>>> end = time.time()
>>> print('median of prevalence :',np.median([pvl[i]  for i, w in enumerate(opt_res.feature_set) if w != 0]))
median of prevalence : 0.5271317829457365

>>> print('total selected feature :',np.sum(opt_res.feature_set))
total selected feature : 226

>>> print("Total cost：%f sec" % (end - start))
Total cost：2.382093 sec
```
![alt](https://github.com/YinchengChen23/PreLect/blob/main/img/img6.png?raw=true)

## Selection profile
To examine the goodness of selected feature set by PreLect, we designed a function `evaluation` to test distinguishment the selected feature set with the classifier which usr-assignment.

`evaluation(x_train, y_train, x_test, y_test, PL_object, classifier)`

|                 |                                                                    |
|-----------------|--------------------------------------------------------------------|
| **Parameters:** | **x_train : *{np-array or sparse matrix} of shape (n_samples, n_features)*** <br> Training dataset.  <br><br> **y_train : *np-array of shape (n_samples, )*** <br> Training label vector. <br><br> **x_test : *{np-array or sparse matrix} of shape (n_samples, n_features)*** <br> Testing dataset.  <br><br> **y_test : *np-array of shape (n_samples, )*** <br> Testing label vector. <br><br> **PL_object : *PreLect class*** <br> The PreLect object which trained with optimized lambda. <br><br> **classifier : *scikit-learn classifier***. <br> A parameterized scikit-learn estimators for examine the performance of feature set selected by PreLect. |
| **Returns:**    | **metrics_dict : *Dict*** <br> A dict of performance measurement {AUC, AUPR, MCC} |

We also developed a function `featureProperty` to get the properties of each features. The user can check the characteristics of the features selected by PreLect.  

`featureProperty(X, y, PL_object)`

|                 |                                                                    |
|-----------------|--------------------------------------------------------------------|
| **Parameters:** | **X : *{pd-dataframe, np-array or sparse matrix} of shape (n_samples, n_features)*** <br> Raw count data. <br><br> y : *{list or np-array} of shape (n_samples, )*** <br> label vector relative to X. <br><br> **PL_object : *PreLect class*** <br> The PreLect object which trained with optimized lambda.         |
| **Returns:**    | **plotdf : *pd-dataframe*** <br> The feature property table.       |

featureProperty(X, y, PL_object):

``` python
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.svm import SVC
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.model_selection import train_test_split

>>> LR = LogisticRegression()
>>> SVM = SVC(probability = True)
>>> RF = RandomForestClassifier()

>>> X_raw = Data_std.to_numpy()
>>> y_array = opt_res.get_y_array(Label)

>>> train_X, test_X, train_y, test_y = train_test_split(X_raw, y_array, test_size=0.2)
>>> perf = evaluation(train_X, train_y, test_X, test_y, opt_res, LR)
>>> print('Logistic Regression :', perf)
Logistic Regression : {'AUC': 0.7518796992481203, 'AUPR': 0.8806141167722912, 'MCC': 0.34684398780964804}

>>> perf = evaluation(train_X, train_y, test_X, test_y, opt_res, SVM)
>>> print('SVM :', perf)
SVM : {'AUC': 0.6090225563909775, 'AUPR': 0.8203639022618696, 'MCC': 0.2218421599939607}

>>> perf = evaluation(train_X, train_y, test_X, test_y, opt_res, RF)
>>> print('RandomForest :', perf)
RandomForest : {'AUC': 0.6165413533834586, 'AUPR': 0.8369912351422049, 'MCC': 0.21804511278195488}

>>> Prop = featureProperty(RawData, Label, opt_res)
>>> Prop['featureID'] = Data_std.columns
>>> Prop
```
![alt](https://github.com/YinchengChen23/PreLect/blob/main/img/img7.png?raw=true)

```python
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
g =sns.scatterplot(x="meanAbundance", y="Variance", hue="select",palette=['r', 'b'], alpha = 0.5, data=Prop)
g.set(yscale="log")
```
![alt](https://github.com/YinchengChen23/PreLect/blob/main/img/img8.png?raw=true)

```python
g =sns.scatterplot(x="prevalence", y="Variance", hue="select", palette=['r', 'b'], alpha = 0.5, data=Prop)
g.set(yscale="log")
```

![alt](https://github.com/YinchengChen23/PreLect/blob/main/img/img9.png?raw=true)

```python
g =sns.scatterplot(x="prevalence_Cancer", y="prevalence_Normal", hue="select", palette=['r', 'b'], data=Prop)
```

![alt](https://github.com/YinchengChen23/PreLect/blob/main/img/img10.png?raw=true)

## Citing Resources
coming soon

## Authors
The following individuals have contributed code to PreLect:
- Yin Cheng, Chen
- Ming Fong, Wu

## Found a Bug
Or would like a feature added? Or maybe drop some feedback?
Just [open a new issue](https://github.com/YinchengChen23/PreLect/issues/new) or send an email to us (yin.cheng.23@gmail.com).
