#!/usr/bin/env python

# Doing indicators function myself because I don't want to import sickitlearn
# from a csv file that is like [tp, tn, fp, fn] with max  50 classes.

import numpy as np
import pandas as pd

from distilbert_classifcation import id2label


def mean(confusion_matrix, i):
    c = confusion_matrix[i]
    return (c[0] + c[1]) / np.sum(c)

def precision(confusion_matrix, i):
    """
    precision or positive predictive value, measure the specificity (probability of a negative result).
    most interesting in our case.
    :param confusion_matrix: confusion matrix as a numpy array, 0: tp, 1:tn, 2: fp, 3, fn
    :param i: label to classify (<50)
    :return: 0 < precision < 1
    """
    c = confusion_matrix[i]
    if c[0] + c[2] == 0: return np.nan
    return c[0] / (c[0] + c[2])


def recall(confusion_matrix, i):
    """
    recall or true positive rate, measure sensitivity (probability of a positive test result).
    :param confusion_matrix: confusion matrix as a numpy array, 0: tp, 1:tn, 2: fp, 3, fn
    :param i: label to classify (<50)
    :return: 0 < recall < 1
    """
    assert i < 50
    c = confusion_matrix[i]
    if c[0] + c[3] == 0: return np.nan
    return c[0] / (c[0] + c[3])


def f1_score(confusion_matrix, i):
    """
    haronic mean of precision and recall, symmetrically represents both.
    :param confusion_matrix:
    :param i:
    :return:
    """
    prec = precision(confusion_matrix, i)
    rec = recall(confusion_matrix, i)
    if prec + rec == 0: return np.nan
    return 2 * prec * rec / (prec + rec)

def print_indicators_i(confusion_matrix, i, tot=True):
    c = confusion_matrix[i]
    print(id2label[i], f" ({i})")
    if tot:
        print("population: ", c[0] + c[3], "total: ", np.sum(c))
    print("------------------------")
    print("tp: ", c[0], "tn: ", c[1])
    print("fp: ", c[2], "fn: ", c[3])
    print("------------------------")
    print("mean:      ", mean(confusion_matrix, i))
    print("precision: ", precision(confusion_matrix, i))
    print("recall:    ", recall(confusion_matrix, i))
    print("f1 score:  ", f1_score(confusion_matrix, i))


def indicators_pandas(confusion_matrix):
    df = pd.DataFrame({"Class": id2label.values(),
                  "mean": [mean(confusion_matrix, j) for j in range(50)],
                  "recall": [recall(confusion_matrix, j) for j in range(50)],
                  "precision": [precision(confusion_matrix, j) for j in range(50)],
                  "f1": [f1_score(confusion_matrix, j) for j in range(50)]})
    return df


if __name__ == "__main__":
    df = pd.read_csv("confusion_matrix_bert.csv")[["tp", "tn", "fp", "fn"]].to_numpy()
    # print(df)
    indicators = indicators_pandas(df)
    indicators.to_csv("indicators_bert.csv")
    print(indicators)

