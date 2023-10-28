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

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from PreLect.core import *

def getPrevalence(X_raw, sample_idx):
    """
    Get the feature prevalence vector according to the given sample set.
        
    Parameters
    ----------
    X_raw : {array-like, sparse matrix {coo, csc, csr}} of shape (n_samples, n_features)
            The original, count data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
    sample_idx : {array-like} of shape (n_samples,)
            The label list relative to X.
            
    Returns
    -------
    prevalence vector : array of shape (n_features,)
    """
    if type(X_raw) is pd.DataFrame:
        X = X_raw.to_numpy()
        X = X[sample_idx]
        pvl_vecter = X.astype(bool).sum(axis=0)/X.shape[0]
    elif type(X_raw) is np.ndarray:
        X = X_raw[sample_idx]
        pvl_vecter = X.astype(bool).sum(axis=0)/X.shape[0]
    elif isspmatrix_csr(X_raw) or isspmatrix_csc(X_raw) or isspmatrix_coo(X_raw):
        X = X_raw[sample_idx]
        pvl_vecter = X.astype(bool).sum(axis=0)/X.shape[0]
        pvl_vecter = pvl_vecter.tolist()[0]
        pvl_vecter = np.array(pvl_vecter)
    else :
        raise ValueError("Unrecognizable data types")
    return pvl_vecter

def stability_measurement(feature_set_collection) :
    """
    Calculates Matthews correlation coefficient to estimate the stability of the selected feature set at different runs.
    
    Parameters
    ----------
    feature_set_collection : list of shape (n_run, n_features)
            The original, count data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
    
    Returns
    -------
    MCC result : array of shape (C(n_run,2),)
    
    References
    ----------
    Jiang, L., Haiminen, N., Carrieri, A. P., Huang, S., Vázquez‐Baeza, Y., Parida, L., ... & Natarajan, L. (2021).
    Utilizing stability criteria in choosing feature selection methods yields reproducible results in microbiome data.
    Biometrics.
    """
    n_selected = feature_set_collection.shape[0]
    i = 0; MCC = []
    while i < n_selected :
        for k in range(i+1, n_selected) :
            mcc_ = matthews_corrcoef(feature_set_collection[i,:], feature_set_collection[k,:])
            MCC.append(mcc_)
        i += 1
    return MCC

def evaluation(x_train, y_train, x_test, y_test, PL_object, classifier):
    """
    Examine the Goodness of selected feature set by user-specified classifier
    
    Parameters
    ----------
    x_train : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
    y_train : array-like of shape (n_samples,)
            Target vector relative to X.
    x_test : {array-like, sparse matrix} of shape (n_samples, n_features)
            Testing vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
    y_test : array-like of shape (n_samples,)
            Testing vector relative to X.
    PL_object : A training PreLect class
    classifier : A scikit-learn classifier.
            A parameterized scikit-learn estimators for examine the performance of feature set selected by PreLect.
    
    Returns
    -------
    dict : a dict of performance measurement {AUC, AUPR, MCC}
    """
    x_subtrain = x_train[:, PL_object.feature_set != 0]
    x_subtest = x_test[:, PL_object.feature_set != 0]
    classifier.fit(x_subtrain, y_train)
    y_pred_proba = classifier.predict_proba(x_subtest)
    y_pred = classifier.predict(x_subtest)
    auroc = roc_auc_score(y_test, y_pred_proba[:, 1])
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
    aupr = metrics.auc(recall, precision)
    mcc = metrics.matthews_corrcoef(y_test, y_pred)
    return {"AUC" : auroc, "AUPR" : aupr, "MCC" : mcc}

def get_data(path):
    from sklearn.datasets import load_svmlight_file
    data = load_svmlight_file(path)
    return data[0], data[1]


def AutoScanning(X_input, X_raw, Y, step=50, device='cpu', training_echo=False,
                 max_iter=10000, tol=1e-4, lr=0.001, alpha=0.9, epsilon=1e-8):
    class_content = np.unique(Y)
    if len(class_content) != 2:
        raise ValueError("This solver needs samples of at only 2 classe.")
    
    n_samples_i, n_features_i = X_input.shape
    n_samples_r, n_features_r = X_raw.shape    
    if n_samples_i != n_samples_r:
        raise ValueError("Found input data with inconsistent numbers of samples with raw data: %r" % [n_samples_i, n_samples_r])
        
    if n_features_i != n_features_r:
        raise ValueError("Found input data with inconsistent numbers of features with raw data: %r" % [n_features_i, n_features_r])
    
    if len(Y) != n_samples_i:
        raise ValueError("Found input label with inconsistent numbers of samples: %r" % [self.n_samples, len(Y)])
        
    if device == 'cuda':
        if not torch.cuda.is_available():
            print("your GPU is not available, PreLect is running with CPU.")
            device= 'cpu'

    y = np.array([0 if yi == class_content[0] else 1 for yi in Y])
    pvl = getPrevalence(X_raw, np.arange(X_raw.shape[0]))
    
    exam_range = [1/(10**i) for i in np.arange(10,-1,-1)]
    select_number = []
    for lmbd in exam_range:
        exam_res = PreLect(lmbd=lmbd, device=device, echo=training_echo, max_iter=max_iter, tol=tol, lr=lr, alpha=alpha, epsilon=epsilon)
        exam_res.fit(X_input, y, pvl)
        select_number.append(np.sum(exam_res.feature_set))
    upper  = np.nan
    for i in range(len(exam_range)):
        if np.isnan(upper):
            if select_number[i] < n_features_i*0.9:
                upper  = exam_range[i]
        if select_number[i] < 10:
            lower  = exam_range[i]
            break
    return np.linspace(np.log(upper), np.log(lower), step)

def LambdaTuning(X_input, X_raw, Y, lmbdrange, k_fold, outdir, device='cpu', training_echo=False,
                  max_iter=10000, tol=1e-4, lr=0.001, alpha=0.9, epsilon=1e-8):
    class_content = np.unique(Y)
    if len(class_content) != 2:
        raise ValueError("This procedure allows only 2 classes.")
    
    n_samples_i, n_features_i = X_input.shape
    n_samples_r, n_features_r = X_raw.shape    
    if n_samples_i != n_samples_r:
        raise ValueError("Found input data with inconsistent numbers of samples with raw data: %r" % [n_samples_i, n_samples_r])
        
    if n_features_i != n_features_r:
        raise ValueError("Found input data with inconsistent numbers of features with raw data: %r" % [n_features_i, n_features_r])
    
    if len(Y) != n_samples_i:
        raise ValueError("Found input label with inconsistent numbers of samples: %r" % [self.n_samples, len(Y)])
    
    if device == 'cuda':
        if not torch.cuda.is_available():
            print("your GPU is not available, PreLect is running with CPU.")
            device= 'cpu'

    if os.path.exists(outdir) == False :
        os.mkdir(outdir)
    os.chdir(outdir)
    
    if type(X_input) is pd.DataFrame:
        X_input = X_input.to_numpy()
    if type(X_raw) is pd.DataFrame:
        X_raw = X_raw.to_numpy()
    y = np.array([0 if yi == class_content[0] else 1 for yi in Y])
    prevalence = getPrevalence(X_raw, np.arange(X_raw.shape[0]))
    n_lambda = len(lmbdrange)
    Z = np.zeros((n_lambda, k_fold, n_features_i), dtype=np.int8)
    
    metrics = ['Percentage', 'Prevalence', 'Feature_number', 'AUC', 'AUPR', 'MCC', 'loss_history', 'error_history', 'pairwiseMCC']
    metrics_dict = dict()
    for m in metrics :
        if m == 'pairwiseMCC' :
            metrics_dict[m] = np.zeros((n_lambda, int(k_fold*(k_fold-1)/2)))
        else : 
            metrics_dict[m] = np.zeros((n_lambda, k_fold))
    
    for i in range(n_lambda):
        start = time.time()
        kfold = StratifiedKFold(n_splits=k_fold, shuffle=True)
        kcount = 0
        for train_ix, test_ix in kfold.split(X_input, y):
            train_X, test_X = X_input[train_ix], X_input[test_ix]
            train_y, test_y = y[train_ix], y[test_ix]
            train_pvl = getPrevalence(X_raw, train_ix)
            lambd = lmbdrange[i]
            examined_lambda = PreLect(lmbd = lambd, device = device, echo = training_echo, max_iter=max_iter, tol=tol, lr=lr, alpha=alpha, epsilon=epsilon)
            examined_lambda.fit(train_X, train_y, train_pvl)
            print(examined_lambda.n_iter_)
            selected_set = examined_lambda.feature_set
            Z[i, kcount, :] = selected_set
            metrics_dict['loss_history'][i,kcount] = examined_lambda.loss_
            metrics_dict['error_history'][i,kcount] = examined_lambda.convergence_
            if np.sum(selected_set) > 1:
                metrics_dict['Feature_number'][i,kcount] = np.sum(selected_set)
                metrics_dict['Percentage'][i,kcount] = np.sum(selected_set)/n_features_i                
                metrics_dict['Prevalence'][i,kcount] = np.median(prevalence[selected_set != 0])
                norm_LR = LogisticRegression(penalty=None)
                perf = evaluation(train_X, train_y, test_X, test_y, examined_lambda, norm_LR)
                metrics_dict['AUC'][i,kcount] = perf['AUC']
                metrics_dict['AUPR'][i,kcount] = perf['AUPR']
                metrics_dict['MCC'][i,kcount] = perf['MCC']
            else:
                metrics_dict['Feature_number'][i,kcount] = 0
                metrics_dict['Percentage'][i,kcount] = 0
                metrics_dict['Prevalence'][i,kcount] = 0
                metrics_dict['AUC'][i,kcount] = 0; 
                metrics_dict['AUPR'][i,kcount] = 0
                metrics_dict['MCC'][i,kcount] = -1
            kcount += 1
        metrics_dict['pairwiseMCC'][i] = stability_measurement(Z[i])
        end = time.time()
        print('lambda is : {lmb}, cost : {tm} min'.format(lmb = lambd, tm = round((end - start)/60, 3)))
    
    metrics_dict['log_lambda_range'] = np.log(lmbdrange)
    for m in list(metrics_dict.keys()) :
        metric_out = str(m) + '.dat'
        np.savetxt(metric_out, metrics_dict[m])
    return metrics_dict

def getTuningResult(result_path):
    os.chdir(result_path)
    measurement = ['Percentage', 'Prevalence', 'Feature_number', 'AUC', 'AUPR', 'MCC', 'loss_history', 'error_history', 'pairwiseMCC', 'log_lambda_range']
    result_dict = dict()
    for m in measurement:
        measurement_in = str(m) + '.dat'
        if measurement_in not in os.listdir():
            raise ValueError("No storage results.")
        res = np.loadtxt(measurement_in, dtype=float)
        result_dict[m] = res
    return result_dict

def LambdaTuningViz(result_dict, metric, savepath=None, fig_width=8, fig_height=4):
    lmbd_range = np.exp(result_dict['log_lambda_range'])
    metrics_recode = result_dict[metric]
    pvl_recode = result_dict['Prevalence']
    m_mean = np.mean(metrics_recode, 1); m_err = np.std(metrics_recode, 1)
    pvl_mean = np.mean(pvl_recode, 1); pvl_err = np.std(pvl_recode, 1)
    pvl_mean = pvl_mean * 100 

    fig, ax1 = plt.subplots(figsize = (fig_width,fig_height))
    ax2 = ax1.twinx()
    ln1 = ax1.errorbar(lmbd_range, m_mean, yerr=m_err, marker='o', c = 'b', linestyle='--', label = metric); ax1.legend(loc='upper left')
    ln2 = ax2.errorbar(lmbd_range, pvl_mean, yerr=pvl_err, marker='o', c = 'r', linestyle='--', label='Prevalence', zorder=1); ax2.legend(loc='upper right')
    ax1.set_xlabel("lambda"); ax1.set_ylabel(metric); ax2.set_ylabel("Prevalence (%)")
    ax1.set(xscale="log")
    if metric in ['Feature_number', 'loss_history', 'error_history']:
        ax1.set(yscale="log")
    if savepath:
        plt.savefig(savepath,dpi =300)
    return fig

def LambdaDecision(result_dict, k, savepath=None, fig_width=8, fig_height=4):
    lmbd_range = np.exp(result_dict['log_lambda_range'])
    loss_recode = result_dict['loss_history']
    pvl_recode = result_dict['Prevalence']
    loss_mean = np.mean(loss_recode, 1); loss_err = np.std(loss_recode, 1)
    pvl_mean = np.mean(pvl_recode, 1); pvl_err = np.std(pvl_recode, 1)
    pvl_mean = pvl_mean * 100 
        
    xs = np.log(lmbd_range); ys = np.log(loss_mean)
    fig, ax1 = plt.subplots(figsize = (fig_width,fig_height))
    ax2 = ax1.twinx()
    dys = np.gradient(ys, xs)
    rgr = DecisionTreeRegressor(max_leaf_nodes = k).fit(xs.reshape(-1, 1), dys.reshape(-1, 1))
    dys_dt = rgr.predict(xs.reshape(-1, 1)).flatten()
    ys_sl = np.ones(len(xs)) * np.nan
    for y in np.unique(dys_dt):
        msk = dys_dt == y
        lin_reg = LinearRegression()
        lin_reg.fit(xs[msk].reshape(-1, 1), ys[msk].reshape(-1, 1))
        ys_sl[msk] = lin_reg.predict(xs[msk].reshape(-1, 1)).flatten()
        ax1.plot(np.exp([xs[msk][0], xs[msk][-1]]), np.exp([ys_sl[msk][0], ys_sl[msk][-1]]), color='r', zorder=5, linewidth = 2)
    
    
    segth = []; count = 0
    for i in range(len(dys_dt)):
        if dys_dt[i] not in segth:
            segth.append(dys_dt[i])
            count += 1
        if count == 1:
            selected_lambda = xs[i]
    
    ax1.errorbar(lmbd_range, loss_mean, marker='o', c='#33CCFF', linestyle='--', label ='BCE loss'); ax1.legend(loc='upper left')
    ax2.errorbar(lmbd_range, pvl_mean, marker='o', c='#FFAA33', linestyle='--', label='Prevalence', zorder=1); ax2.legend(loc='upper right')
    ax1.set(xscale="log"); ax1.set(yscale="log")
    ax1.set_xlabel("lambda"); ax1.set_ylabel("loss"); ax2.set_ylabel("Prevalence (%)")
    selected_lambda = np.exp(selected_lambda); plt.axvline(x=selected_lambda, color = 'black', linestyle=':')
    plt.show()
    if savepath:
        plt.savefig(savepath,dpi =300)
    return selected_lambda, fig

def scipySparseVars(a, axis=None):
    a_squared = a.copy()
    a_squared.data **= 2
    return a_squared.mean(axis) - np.square(a.mean(axis))

def featureProperty(X, y, PL_object):
    if type(X) is pd.DataFrame:
        X = X.to_numpy()
    y = PL_object.get_y_array(y)
    
    if type(X) is np.ndarray:
        nonzeroSamples = X.sum(axis=1) != 0
        X_ = X[nonzeroSamples,:]
        y_ = y[nonzeroSamples]
        
        XF = X_/X_.sum(axis=1)[:,None]
        RA = XF.mean(axis = 0)
        Var = X_.std(axis = 0)**2

    elif isspmatrix_csr(X) or isspmatrix_csc(X) or isspmatrix_coo(X):
        rowSum = X.sum(axis=1).reshape(-1)
        nonzeroSamples = np.array(rowSum != 0)[0]
        X_ = X[nonzeroSamples,:]
        y_ = y[nonzeroSamples]
        RA = X_.mean(axis = 0); RA = np.array(RA).reshape(-1) #XF
        Var = scipySparseVars(X_, axis = 0); Var = np.array(Var).reshape(-1)
        
    else :
        raise ValueError("X is unrecognizable data types")
        
    selection = ["PreLect" if i == 1 else "No selected" for i in PL_object.feature_set]
    classIdx = {v: k for k, v in PL_object.classes_.items()}
    class0Idx = np.array([i for i, la in enumerate(y_) if la == 0])
    class1Idx = np.array([i for i, la in enumerate(y_) if la == 1])
    wholePvl = getPrevalence(X_, np.arange(X_.shape[0]))
    class0Pvl = getPrevalence(X_, class0Idx)
    class1Pvl = getPrevalence(X_, class1Idx)
    C0head = 'prevalence_' + str(list(PL_object.classes_.keys())[0])
    C1head = 'prevalence_' + str(list(PL_object.classes_.keys())[1])
    plotdf = pd.DataFrame({'meanAbundance' : RA,'Variance' : Var, 'select' : selection, 'prevalence' : wholePvl, C0head : class0Pvl, C1head : class1Pvl})
    return plotdf