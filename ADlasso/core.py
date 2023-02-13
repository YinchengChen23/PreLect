import numpy as np
import pandas as pd
import time
import os
import scipy
from scipy.sparse import *
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn

class ADlasso():
    """
    Logistic Regression with Adapted LASSO
    min_{w} sum_{i}^{n}{ BCE(y_i, sigmoid(X_i x w + b)) } + lambda x sum_{j}^{d} ||frac{w_j}{p_j}||
    
    Parameters
    ----------
    lmbd : float, default=1.0e-5
        Regularization intensity, lambda.
        
    max_iter : int, default=200,000
        Maximum number of iterations for the solver.
        
    tol : float, default=1e-4
        Tolerance for stopping criteria.
        
    lr : float, default=0.001
        Learning rate, to regular the step size in RMSprop.
        
    alpha : float, default=0.9
        The proportion with reference to gradient history in RMSprop.
        
    epsilon : float, default=1e-8
        A small value to avoid the denominator being zero
        
    device : {'cup', 'cuda'}, default='cup'
        Specify the running device.
        
    Attributes
    ----------
    classes_ : dict
        A dictionary of class labels corresponding to binary prediction. 
    n_iter_ : int
        The iteration number after optimization.
    loss_ : float
        The loss value after optimization.
    convergence_ : float
        The degree of convergence after optimization.
    w : ndarray of shape (n_features, )
        The weight of the features in the decision function.
    b : ndarray of shape (1, )
        The bias in the decision function.
    feature_set : ndarray of shape (n_features, )
        The list to indicate the selected features, 1 is selected while 0 is not.
    feature_sort : ndarray of shape (n_features, )
        The list to sort the important features with index.

    """
    def __init__(self, lmbd=1e-5, max_iter=10000, tol=1e-4, lr=0.001, alpha=0.9, epsilon=1e-8, device='cpu', echo=False):
        super().__init__()
        self.lmbd = lmbd
        self.max_iter = int(max_iter)
        self.tol = tol
        self.lr = lr
        self.alpha = alpha
        self.epsilon = epsilon
        self.sigmoid = nn.Sigmoid()
        self.BCE = nn.BCELoss()
        self.UsingDevice = device
        self.echo = echo
        self.classes_ = None
        self.n_iter_ = None
        self.loss_ = None
        self.convergence_ = None
        self.w = None
        self.b = None
        self.feature_set = None
        self.feature_sort = None
        
    def np2tensor(self, X):
        X = torch.from_numpy(X).float()
        self.n_samples, self.n_features = X.shape
        bias = torch.ones(self.n_samples).reshape(-1, 1) # create a all 1 vector for bias
        X_data = torch.cat((X, bias), 1).to(self.UsingDevice)        # append the all 1 column into X for representing bias
        return X_data, X_data.T
    
    def coo2sparse_tensor(self, X):
        self.n_samples, self.n_features = X.shape
        bias = np.ones(self.n_samples).reshape(-1, 1)  # create a all 1 vector for bias
        X = scipy.sparse.hstack((X, bias))                         # append the all 1 column into X for representing bias
        values = X.data; v = torch.FloatTensor(values)
        indices = np.vstack((X.row, X.col)); i = torch.LongTensor(indices)
        Xshape = X.shape
        X_data = torch.sparse_coo_tensor(i, v, size=Xshape, dtype=torch.float32, requires_grad=False).to(self.UsingDevice) 
        X_dataT = torch.transpose(X_data, 0, 1).to(self.UsingDevice)
        return X_data, X_dataT
        
    def initialize(self, X, Y, pvl):
        class_content = np.unique(Y)
        if len(class_content) != 2:
            raise ValueError("This solver needs samples of at only 2 classes, try to use MultiClassADlasso.")
        
        if self.UsingDevice not in ['cpu','cuda']:
            raise ValueError("Wrong device assignment.")

        if self.UsingDevice == 'cuda':
            if not torch.cuda.is_available():
                print("your GPU is not available, ADlasso is running with CPU.")
                self.UsingDevice = 'cpu'
        
        if type(X) is np.ndarray:
            X, XT = self.np2tensor(X)
        elif type(X) is pd.DataFrame:
            X, XT = self.np2tensor(X.to_numpy())
        elif isspmatrix_csr(X):
            X, XT = self.coo2sparse_tensor(X.tocoo())
        elif isspmatrix_csc(X):
            X, XT = self.coo2sparse_tensor(X.tocoo())
        elif isspmatrix_coo(X):
            X, XT = self.coo2sparse_tensor(X)
        else :
            raise ValueError("X is unrecognizable data type")
            
        if len(Y) != self.n_samples:
            raise ValueError("Found input label with inconsistent numbers of samples: %r" % [self.n_samples, len(Y)])

        y = np.array([0 if yi == class_content[0] else 1 for yi in Y])
        y = torch.from_numpy(y).float().reshape(self.n_samples, 1).to(self.UsingDevice)
        self.classes_ = {class_content[0] : 0, class_content[1] : 1}
        
        if pvl.shape[0] != self.n_features:
            raise ValueError("Found input prevalence vector with inconsistent numbers of features: %r" % [self.n_features, pvl.shape[0]])
        pvl = np.append(pvl,1)     # append 1 for representing bias prevalence
        pvl = torch.from_numpy(pvl).float().reshape(-1, 1).to(self.UsingDevice)
        
        weight = torch.zeros(self.n_features+1, requires_grad = False).reshape(-1, 1).to(self.UsingDevice)
        return X, XT, y, pvl, weight
        
    def logistic_gradient(self, X, XT, y, w):
        diff = self.sigmoid(X.mm(w)) - y
        gradient = XT.mm(diff)/self.n_samples
        return gradient
        
    def RMSprop(self, w, prev_v, gradient, pvl):
        v = self.alpha*(prev_v.square()) + ((1 - self.alpha)*(gradient.square())) + self.epsilon
        v = v.sqrt_()
        z = w - (self.lr/v)*pvl*gradient
        return z, v

    def proximal_GD(self, X, XT, y, w, v, pvl, thres):
        gradient = self.logistic_gradient(X, XT, y, w)
        z, v_ = self.RMSprop(w, v, gradient, pvl)
        w_ = z.sign()*(z.abs() - thres).clamp(min=0)
        w_[self.n_features, 0] = z[self.n_features, 0]
        return w_, v_
        
    def fit(self, X_input, Y, prevalence):
        """
        Fit the model according to the given training data.
        
        Parameters
        ----------
        X_input : {array-like, sparse matrix {coo, csc, csr}} of shape (n_samples, n_features)
            The normalized or transformed data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : {array-like, list} of shape (n_samples,)
            The label list relative to X.
        prevalence : array of shape (n_features,)
            The prevalence vector relative to each feature in X.
        echo : bool
            If True, will print the final training result.
        
        Returns
        -------
        self
            Fitted estimator.
        """
        X, Xt, y, prevalence, current_w = self.initialize(X_input, Y, prevalence)
        current_v = current_w
        thres = self.lmbd/prevalence
        minimum_loss = self.BCE(self.sigmoid(X.mm(current_w)), y)
        mini_iter = 0; mini_loss = 1e+10; mini_diff = 1e+10
        for i in range(self.max_iter):
            prev_w = current_w
            current_w, current_v = self.proximal_GD(X, Xt, y, current_w, current_v, prevalence, thres)
            lossvalue = self.BCE(self.sigmoid(X.mm(current_w)), y).item()
            diff_w = torch.sqrt_(torch.sum(torch.square_(current_w - prev_w)))
            if diff_w <= self.tol:
                mini_iter = i; mini_loss = lossvalue; mini_diff = diff_w; best_w = current_w
                break
            if mini_loss > lossvalue:
                mini_iter = i; mini_loss = lossvalue; mini_diff = diff_w; best_w = current_w

        if self.UsingDevice == 'cuda':
            best_w = best_w.cpu().numpy().reshape(-1)
        else:
            best_w = best_w.numpy().reshape(-1)
        
        self.n_iter_ = mini_iter; self.loss_ = mini_loss; self.convergence_ = mini_diff.item()

        if self.echo:
            print('minimum epoch = ', self.n_iter_, '; minimum lost = ', self.loss_, '; diff weight = ', self.convergence_)

        self.w = best_w[0:self.n_features]; self.b = best_w[self.n_features]
        self.feature_set = np.where(self.w != 0, 1, 0)
        weight_abs = np.abs(self.w)
        self.feature_sort= np.argsort(-weight_abs)
        
    def predict_proba(self, X):
        """
        Probability estimates.
        
        Parameters
        ----------
        X : {array-like, sparse matrix {coo, csc, csr}} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        
        Returns
        -------
        pred_proba : array of shape (n_samples,)
            Returns the probability of the sample for binary class in the model,
            where classes are show in ``self.classes_``.
        """
        z = np.exp(-(X.dot(self.w)+self.b))
        pred_proba = 1 / (1 + z)
        return pred_proba
    
    def predict(self, X):
        """
        Prediction.
        
        Parameters
        ----------
        X : {array-like, sparse matrix {coo, csc, csr}} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        
        Returns
        -------
        pred : array of shape (n_samples,)
            Returns the prediction result for binary classifier,
            where classes are show in ``self.classes_``.
        """
        z = np.exp(-(X.dot(self.w)+self.b))
        pred_proba = 1 / (1 + z)
        pred = np.where(pred_proba > 0.5, 1, 0)
        return pred
    
    def get_y_array(self, label_list):
        """
        Get the corresponding label array in this model.
        
        Parameters
        ----------
        label_list : list of shape (n_samples)
        
        Returns
        -------
        y : array of shape (n_samples,)
        """
        return  np.array([self.classes_[yi] for yi in label_list])
    
    def score(self, X, y):
        """
        Goodness of fit estimation.
        
        Parameters
        ----------
        X : {array-like, sparse matrix {coo, csc, csr}} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : array of shape (n_samples,)
        
        Returns
        -------
        score : dictionary of measurement (AUC, AUPR, MCC, precision, recall)
        """
        y_pred_proba = self.predict_proba(X)
        y_pred = self.predict(X)
        y_true = self.get_y_array(y)
        auroc = roc_auc_score(y_true, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        aupr = metrics.auc(recall, precision)
        mcc = matthews_corrcoef(y_true, y_pred)
        return {"AUC" : auroc, "AUPR" : aupr, "MCC" : mcc, "Precision" : precision, "Recall" : recall}

    def writeList(self, outpath=None, featureNameList=None):
        """
        Export the selection result.
        
        Parameters
        ----------
        outpath : str
                  Absolute path for output file.
        featureNameList : list or array of shape (n_features,)
                  A list contains feature name.
        
        Returns
        -------
        File : first column : Name or index of selected feature.
               second column : Weight of each feauture.
               third column : Tendency of each feature.
        """
        if featureNameList is not None:
            if len(self.feature_set) != len(featureNameList):
                raise ValueError("Found input feature list with inconsistent numbers of features: %r" % [len(self.feature_set), len(featureNameList)])

        dirpath = os.path.dirname(outpath)
        if not dirpath:
            raise ValueError("The folder you assigned does not exist.")

        classes = {v: k for k, v in self.classes_.items()}
        w = open(outpath,'w')
        for ix, wi in enumerate(self.w):
            if wi != 0:
                featureID = featureNameList[ix] if featureNameList is not None else ix
                tendency = classes[0] if wi < 0 else classes[1]
                w.writelines(featureID + "\t" + str(wi) + "\t" + tendency + '\n')
        w.close()
