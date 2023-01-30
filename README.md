# **Adapted LASSO : A Robust Approach for Feature Engineering in Sparse Matrices**


[![License](https://img.shields.io/badge/License-MIT-blue)](#license)


Table of Contents
====================
- [Installation](#installation)
- [General usage](#general-usage)
- [Lambda tuning](#lambda-tuning)
- [Selection profile](#selection-profile)
- [Citing Resources](#citing-resources)
- [Found a Bug](#found-a-bug)

## Installation

  [Pytorch](https://pytorch.org/) is essential, please install it first.

  Install ADlasso via package clone from GitHub repository.

```Shell
$ git clone https://github.com/YinchengChen23/ADlasso
$ cd ADlasso
$ pip install . 
```

## General usage
ADlasso is a classifier based on logistic regression with an L1-like penalty, designed with a scikit-learn-like API. We can set the parameters using the `ADlasso` class and use the built-in function `.fit()` to fit the model to the training data. Eventually, the selection results are presented in the `.feature_set` and `.feature_sort` attributes.

```python
class ADlasso(lmbd=1e-5, max_iter=10000, tol=1e-4, lr=0.001, alpha=0.9, epsilon=1e-8,
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
| device   | String  | cpu      | `cpu` : run with CPU; `cuda` : run with GPU.                     |
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
>>> import numpy as np
>>> import pandas as pd
>>> import ADlasso
>>> from ADlasso.AD_utils import *

>>> res = ADlasso(lmbd=1e-4)
>>> res.fit(train_X, train_y, pvl_vec)
```
<br>

> **WARNING**
> Since ADlasso has not yet been established as a full scikit-learn API, so in the process of using it, you must first declare a class before doing the fit, you cannot do the fit directly.


```python
# Wrong way
>>> res = ADlasso(lmbd=1e-5,).fit(train_X, train_y, pvl_vec) 
# Correct way
>>> res = ADlasso(lmbd=1e-5,)
>>> res = ADlasso(lmbd=1e-5,).fit(train_X, train_y, pvl_vec)
```
#### Methods
> get_y_array(label_list)

Get the corresponding label array in this model.






| Functionn                    | Description                                         |
|------------------------------|-----------------------------------------------------|
| get_y_array(label_list)      | Get the corresponding label array in this model.    |
| fit(X, y, pvl)               | Fit the model according to the given training data. |
| predict_proba(X)             | Probability estimates.                              |
| predict(X)                   | Predict class labels for samples in X.              |
| score(X, y)                  | Return the metrics Including {AUC, AUPR, MCC, precision, recall} on the given test data and labels. |

> get_y_array(label_list) 
<id="First">









## Lambda tuning

## Selection profile
## Citing Resources
coming soon
## Found a Bug
Or would like a feature added? Or maybe drop some feedback?
Just [open a new issue](https://github.com/ChongLC/MinimalSetofViralPeptidome-UNIQmin/issues/new) or send an email to us (yin.cheng.23@gmail.com).