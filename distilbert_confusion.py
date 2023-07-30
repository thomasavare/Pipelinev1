#!/usr/bin/env python

# Confusion matrix of just the text classification model on the same dataset

import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

from distilbert_classifcation import load_bert, classify_id


def true_pos(res, gt, i):
    return np.sum(np.logical_and(gt == i, res == i))


def true_neg(res, gt, i):
    return np.sum(np.logical_and(gt != i, res != i))


def false_pos(res, gt, i):
    return np.sum(np.logical_and(gt != i, res == i))


def false_neg(res, gt, i):
    return np.sum(np.logical_and(gt == i, res != i))


def confusion_matrix(res, gt, num_class):
    confusion = np.zeros((num_class, 4))
    for j in range(num_class):
        confusion[j] = [true_pos(res, gt, j), true_neg(res, gt, j), false_pos(res, gt, j),
                        false_neg(res, gt, j)]
    return confusion


if __name__ == "__main__":
    size = 100
    # first, let's download audiodataset
    dataset = load_dataset("thomasavare/waste-classification-audio", split=f"train[:{size}]")

    # Load whisper and bert elements
    tokenizer, cls_model = load_bert()

    res = np.zeros(size)

    for i in tqdm(range(size)):
        res[i] = classify_id(dataset["translation"][i], tokenizer=tokenizer, model=cls_model)
        print(dataset["translation"][i], res[i], dataset["Class_index"][i], res[i] == dataset["Class_index"][i])

    conf = confusion_matrix(res, np.array(dataset["Class_index"]), 50)
    print(conf)

    df_conf = pd.DataFrame(conf, columns=["tp", "tn", "fp", "fn"])
    df_conf.to_csv("confusion_matrix_bert.csv")