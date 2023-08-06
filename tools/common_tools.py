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


def label_map(labels):
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
    inverse_mapping = dict((v,k) for k, v in mapping.items())
    mapping_dict = {'mapping':mapping,'inverse_mapping':inverse_mapping}
    return mapping_dict


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
 

def split_df(df, split_size, random_state):
    train, valid = train_test_split(df, 
                                    test_size=split_size,
                                    random_state=random_state, 
                                    stratify=df.label)
    return train, valid


def adjust_df(df):
    mapping = {'std_name':'text','adname_1':'province',
               'adname_2':'city','urban_tag':'urban'}
    df = df.reset_index().rename(columns=mapping)
    return df


def class_report(true_labels, predicted, label_map):
    test_result = classification_report(true_labels, 
                                        predicted, 
                                        target_names=list(label_map.values()),
                                        output_dict=True)
    res_tb = pd.DataFrame(test_result).transpose().apply(lambda x:round(x,4))
    return res_tb


def release(pre, res):
    
    pre = add_info(adjust_df(pre)).set_index(['shop_id','sa_id'])[['text','source_type']]
    res = res.reset_index().set_index(['shop_id','sa_id'])
    final = pre.align(res, join='outer', axis=0)[1].fillna(0)
    final = final.set_index(['source_type','text'], append=True)
    final = final.droplevel('text')
    assert len(final[final.isna().any(axis=1)])==0, 'contains nans'
    assert len(final)==len(pre), 'unequal len'
    return final


if __name__ == '__main__':
    pass