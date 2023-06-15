# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 00:08:35 2023

@author: nick
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from itertools import product
import torch
from sklearn.metrics import classification_report


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
 

def split_df(data, test_size, random_state):
    train, valid = train_test_split(data, 
                                    test_size=test_size,
                                    random_state=random_state, 
                                    stratify=data.label)
    return train, valid


def class_report(true_labels, predicted, label_map):
    test_result = classification_report(true_labels, 
                                        predicted, 
                                        target_names=list(label_map.values()),
                                        output_dict=True)
    res_tb = pd.DataFrame(test_result).transpose().apply(lambda x:round(x,4))
    return res_tb


def release(pre_cleaned, pred):
    pre_cleaned = pre_cleaned.set_index(['shop_id', 'sa_id', 'source_type', 'text'])
    indexes = (set(pre_cleaned.index.get_level_values('shop_id'))-
               set(pred.index.get_level_values('shop_id')))
    result = (pre_cleaned.loc[pre_cleaned.index.get_level_values('shop_id').
                              isin(indexes)])
    result = result.drop(result.columns,axis=1)
    
    final = pd.concat([result,pred], axis=0).fillna(0)
    # final.loc[final.index.get_level_values('source_type').isin(['comment_baby', 'comment_cos']), :] = 0
    final = final.droplevel('text')
    assert len(final[final.isna().any(axis=1)])==0, 'contains nans'
    assert len(final)==len(pre_cleaned), 'unequal len'
    return final


if __name__ == '__main__':
    pass