import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score, confusion_matrix, recall_score
from river import base
import math
from collections import deque
import sys

threshold= None
f1_recall = None

def pr_auc(labels, scores, threshold, f1_recall):
    precision, recall, _ = precision_recall_curve(labels, scores)
    return auc(recall, precision)


def roc_auc(labels, scores, threshold, f1_recall):
    fpr, tpr, _ = roc_curve(labels, scores)
    return auc(fpr, tpr)


def f1_prec_rec_thresh(labels, scores):
    precision, recall, threshold = precision_recall_curve(labels, scores)
    count = 2 * precision * recall
    denom = precision + recall
    f1 = np.divide(count, denom, out=np.zeros_like(count), where=denom != 0)
    return f1, precision, recall, threshold


def max_f1_threshold(labels, scores):
    f1, _, _, threshold = f1_prec_rec_thresh(labels, scores)
    return threshold[np.argmax(f1)]

def binary_scores(scores, threshold):
    binary_array = []
    for score in scores:
        if score >= threshold:
            binary_array.append(1)
        else:
            binary_array.append(0)
    return binary_array

def f1_recall_score(labels, scores, threshold):
    scores = binary_scores(scores, threshold)
    f1= f1_score(labels, scores, average=None)
    recall = recall_score(labels, scores, average=None)
    return f1, recall


def f1_0(labels, scores, threshold, f1_recall):
    return f1_recall[0][0]

def f1_1(labels, scores, threshold, f1_recall):
    return f1_recall[0][1]

def recall_0(labels, scores, threshold, f1_recall):
    return f1_recall[1][0]

def recall_1(labels, scores, threshold, f1_recall):
    return f1_recall[1][1]

def geo_mean(labels, scores, threshold, f1_recall):

    tn, fp, fn, tp = confusion_matrix(labels, [1 if s >= threshold else 0 for s in scores]).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    g_mean = np.sqrt(sensitivity * specificity)

    return g_mean

def compute_rates(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)###
    precision, recall, _ = precision_recall_curve(labels, scores)
    return fpr, tpr, recall, precision

METRICS = {
    "PR-AUC": pr_auc,
    "ROC-AUC": roc_auc,
    "F1_0": f1_0,
    "F1_1": f1_1,
    "Recall_0": recall_0,
    "Recall_1": recall_1,
    "Geo-Mean": geo_mean,
}

def calculate_object_size(obj, unit='byte'):

    seen = set()
    to_visit = deque()
    byte_size = 0

    to_visit.append(obj)

    while True:
        try:
            obj = to_visit.popleft()
        except IndexError:
            break

        # If element was already covered, skip it
        if id(obj) in seen:
            continue

        # Update size accounting
        byte_size += sys.getsizeof(obj)

        # Mark element as seen
        seen.add(id(obj))

        # Add keys and values for size account
        if isinstance(obj, dict):
            for v in obj.values():
                to_visit.append(v)

            for k in obj.keys():
                to_visit.append(k)
        elif hasattr(obj, '__dict__'):
            to_visit.append(obj.__dict__)
        elif hasattr(obj, '__iter__') and \
                not isinstance(obj, (str, bytes, bytearray)):
            for i in obj:
                to_visit.append(i)

    if unit == 'kB':
        final_size = byte_size / 1024
    elif unit == 'MB':
        final_size = byte_size / (2 ** 20)
    else:
        final_size = byte_size

    return final_size

def compute_metrics(labels, scores, metrics=METRICS):
    
    result = {}
    threshold = max_f1_threshold(labels, scores)
    f1_recall = f1_recall_score(labels, scores, threshold)

    for name, metric in metrics.items():
        result[name] = metric(labels, scores, threshold, f1_recall)

    for key in result:
        if math.isnan(result[key]):
            result[key] = 0

    return result