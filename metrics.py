'''
Evaluation metrics functions.
'''
import datetime
import json
import os
from typing import Tuple

import numpy
# import math
import numpy as np
import collections

# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve, auc
# from sklearn.metrics import average_precision_score
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from scipy.stats import rankdata
from matplotlib import pyplot


def _retype(y_prob, y):
    if not isinstance(y, (collections.Sequence, np.ndarray)):
        y_prob = [y_prob]
        y = [y]
    y_prob = np.array(y_prob)
    y = np.array(y)

    return y_prob, y


def _binarize(y, n_classes=None):
    return label_binarize(y, classes=range(n_classes))


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(y_prob, y, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    predicted = [np.argsort(p_)[-k:][::-1] for p_ in y_prob]
    actual = [[y_] for y_ in y]
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def mean_rank(y_prob, y):
    ranks = []
    n_classes = y_prob.shape[1]
    for p_, y_ in zip(y_prob, y):
        ranks += [n_classes - rankdata(p_, method='max')[y_]]

    return sum(ranks) / float(len(ranks))


def hits_k(y_prob, y, k=10):
    acc = []
    for p_, y_ in zip(y_prob, y):
        top_k = p_.argsort()[-k:][::-1]
        acc += [1. if y_ in top_k else 0.]
    return sum(acc) / len(acc)


# def roc_auc(y_prob, y):
#     y = _binarize(y, n_classes=y_prob.shape[1])
#     fpr, tpr, _ = roc_curve(y.ravel(), y_prob.ravel())
#     return auc(fpr, tpr)


# def log_prob(y_prob, y):
#     scores = []
#     for p_, y_ in zip(y_prob, y):
#         assert abs(np.sum(p_) - 1) < 1e-8
#         scores += [-math.log(p_[y_]) + 1e-8]
#         print p_, y_

#     return sum(scores) / len(scores)


def portfolio(y_prob, y, k_list=None):
    y_prob, y = _retype(y_prob, y)
    # scores = {'auc': roc_auc(y_prob, y)}
    # scores = {'mean-rank:': mean_rank(y_prob, y)}
    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = hits_k(y_prob, y, k=k)
        scores['map@' + str(k)] = mapk(y_prob, y, k=k)

    return scores


def calc_metrics(output, target, seq, seq_len, train_nodes, log):
    precision_values = []
    recall_values = []
    f1_values = []
    fpr_values = []
    batch_size = output.shape[0]
    target_lengths = np.count_nonzero(target, axis=1)
    new_target_counts = np.zeros(batch_size, np.int32)
    for i in range(batch_size):
        new_target_counts[i] = np.sum(np.logical_not(np.isin(target[i, :target_lengths[i]], train_nodes)))
    ref = train_nodes.size - np.count_nonzero(seq, axis=1) + new_target_counts
    # log.info(f"target_lengths = {target_lengths}")
    # log.info(f"new_target_counts = {new_target_counts}")
    # log.info(f"ref = {ref}")

    for k in range(1, seq_len + 1):
        tp = np.zeros(batch_size, dtype=np.float32)
        for i in range(batch_size):
            tp[i] = np.intersect1d(output[i, :k], target[i, :target_lengths[i]]).size
        precision = tp / k
        recall = np.divide(tp, target_lengths)
        f1 = 2 * np.multiply(precision, recall) / (precision + recall)
        f1[np.isnan(f1)] = 0
        fpr = np.divide(np.count_nonzero(output[:, :k], axis=1) - tp, ref)
        # log.info(f"k = {k}")
        # log.info(f"tp = {tp}")
        # log.info(f"precision = {precision}")
        # log.info(f"recall = {recall}")
        # log.info(f"f1 = {f1}")
        # log.info(f"fpr = {fpr}")

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        fpr_values.append(fpr)

    precision_values = np.array(precision_values)
    recall_values = np.array(recall_values)
    f1_values = np.array(f1_values)
    fpr_values = np.array(fpr_values)

    return precision_values, recall_values, f1_values, fpr_values


def auc_roc(fprs: list, tprs: list):
    """ area under curve of ROC """
    fprs, tprs = prepare_roc(fprs, tprs)
    return metrics.auc(fprs, tprs)


def prepare_roc(fprs, tprs) -> Tuple[np.array, np.array]:
    """ Preprocess fpr and tpr values and sort them to calculate auc_roc or to plot ROC """
    # Every ROC curve must have 2 points <0,0> (no output) and <1,1> (returning all reference set as output).
    fprs, tprs = np.array(fprs), np.array(tprs)
    indexes = fprs.argsort()
    fprs = fprs[indexes]
    tprs = tprs[indexes]
    if (fprs[0], tprs[0]) != (0, 0):
        fprs = np.hstack((np.array([0]), fprs))
        tprs = np.hstack((np.array([0]), tprs))
    if (fprs[-1], tprs[-1]) != (1, 1):
        fprs = np.hstack((fprs, np.array([1])))
        tprs = np.hstack((tprs, np.array([1])))
    return fprs, tprs


def save_roc(fpr_list: list, tpr_list: list, dataset: str):
    """
    Save ROC plot as png and FPR-TPR values as json.
    """
    fpr, tpr = prepare_roc(fpr_list, tpr_list)
    pyplot.figure()
    pyplot.plot(fpr, tpr)
    pyplot.axis((0, 1, 0, 1))
    pyplot.xlabel("fpr")
    pyplot.ylabel("tpr")
    results_path = 'results'
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    base_name = f'{dataset}-deepdiffuse-nodes-roc-{datetime.datetime.now()}'.replace(" ", "-")
    pyplot.savefig(os.path.join(results_path, f'{base_name}.png'))
    # pyplot.show()
    with open(os.path.join(results_path, f'{base_name}.json'), "w") as f:
        json.dump({"fpr": fpr.tolist(), "tpr": tpr.tolist()}, f)
