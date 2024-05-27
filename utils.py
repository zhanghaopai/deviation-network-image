#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Guansong Pang
The algorithm was implemented using Python 3.6.6, Keras 2.2.2 and TensorFlow 1.10.1.
More details can be found in our KDD19 paper.
Guansong Pang, Chunhua Shen, and Anton van den Hengel. 2019. 
Deep Anomaly Detection with Deviation Networks. 
In The 25th ACM SIGKDDConference on Knowledge Discovery and Data Mining (KDD ’19),
August4–8, 2019, Anchorage, AK, USA.ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3292500.3330871
"""
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
import numpy as np


def aucPerformance(mse, labels, prt=True):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    if prt:
        print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap

def fpr_and_fnr_performance(y_true, y_score):
    _, _,thresholds =roc_curve(y_true, y_score)
    balance_fnr=-0.1
    balance_fpr=-0.1
    min=1000000.0
    index=0
    for i, threshold in enumerate(thresholds):
        # 误报率
        fnr = _calculate_fnr(y_true, y_score, threshold)
        # 漏报率
        fpr=_calculate_fpr(y_true, y_score, threshold)
        if fpr + fnr < min:
            min=fnr+fpr
            balance_fpr = fpr
            balance_fnr = fnr
            index = i

    print("误报率最好为: {}，漏报率最好为：{}，此时分数阈值为：{}\n".format(balance_fnr, balance_fpr, thresholds[index]))



def _calculate_fpr(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int) # 分数大于异常阈值的算作异常，标为1
    false_positives = np.sum((y_true == 1) & (y_pred == 0))
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))
    if false_positives + true_negatives > 0:
        fpr = false_positives / (false_positives + true_negatives)
    else:
        fpr = 0  # 避免除以零的情况
    return fpr

def _calculate_fnr(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_negatives = np.sum((y_true == 0) & (y_pred == 1))
    if true_positives + false_negatives > 0:
        fnr = false_negatives / (true_positives + false_negatives)
    else:
        fnr = 0  # 避免除以零的情况
    return fnr

