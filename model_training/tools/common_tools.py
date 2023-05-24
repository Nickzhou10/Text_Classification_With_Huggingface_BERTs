# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 00:08:35 2023

@author: nick
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from itertools import product
import torch


def label_to_id(labels):
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    transformed = label_encoder.transform(labels)
    mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
    return transformed, mapping


def add_info(df):
    df.urban = df.urban.replace({'Y': '市区', 'N': '郊区'})
    df.text_bk = df.text.copy()
    df.text = df.province + df.city + df.urban + df.text 
    return df


def generate_combinations(df, columns):
    new_rows = []
    for row in df.itertuples(index=False):
        combinations = product(*[getattr(row, col) for col in columns])
        for combo in combinations:
            new_rows.append(combo)
    new_df = pd.DataFrame(new_rows, columns=columns)
    return new_df


def compute_class_weights(targets, smoothing_factor=0.0):
    class_counts = torch.bincount(targets)
    total_samples = torch.sum(class_counts)
    class_weights = 1.0 / (class_counts + smoothing_factor)
    class_weights = class_weights * total_samples / torch.sum(class_weights)
    return class_weights





if __name__ == '__main__':
    pass