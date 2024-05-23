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

from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np


def aucPerformance(mse, labels, prt=True):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    if prt:
        print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap

def fpr_performance(y_true, y_pred):
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))
    if false_positives + true_negatives > 0:
        fpr = false_positives / (false_positives + true_negatives)
    else:
        fpr = 0  # 避免除以零的情况
    print("False Positive Rate (误报率): {:.4f}".format(fpr))
    return fpr

def fnr_performance(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    if true_positives + false_negatives > 0:
        fnr = false_negatives / (true_positives + false_negatives)
    else:
        fnr = 0  # 避免除以零的情况

    print("False Negative Rate (漏报率): {:.4f}".format(fnr))
    return fnr
