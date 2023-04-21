# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 11:42:37 2023
Precision TP/(TP+FP) 低=假阳多=较多原本不是此类的变成此类
recall TP/(TP+FN) 低=假阴多=较多原本是此类的变成非此类
@author: zhni2001
"""
import pandas as pd
import numpy as np
from clean_tools.data_chker import checker
from clean_tools.data_cleaner import cleaner
from sklearn.model_selection import train_test_split

def process_raw(data_path):
    col_mapping = {'sa_id':'store_code','std_name':'text','adname_1':'province',
                   'adname_2':'city','nls_store_type':'label','fix_name':'store_name',
                   'source_type':'source','urban_tag':'urban'}
    col_list = ['sa_id', 'std_name', 'adname_1', 'adname_2','urban_tag','nls_store_type','source_type']
    store_train = (pd.read_pickle(data_path+'2206_train_data.pickle'))    
    store_train = store_train.loc[store_train['nls_store_type']!='cos'].copy()
    store_train = store_train[col_list].rename(columns=col_mapping)
    
    others = (pd.read_pickle(data_path +'202206_train_data_others.pickle')).assign(nls_store_type='others')
    others['urban_tag'].fillna('N',inplace=True)
    others = others[col_list].rename(columns=col_mapping)
    others = others.drop_duplicates('store_code')
    raw_data = pd.concat([store_train,others],axis=0)
    return raw_data 


def split_data(data):
    train_valid, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data.label)
    train, valid = train_test_split(train_valid, test_size=0.125, random_state=42, stratify=train_valid.label)
    return train, valid, test


if __name__ == '__main__':
    data_path = 'E:/nlp_models/sa_current/LSTM/2206/name2channel/data/'
    output_path = 'E:/nlp_models/test_data/SA_store_type/release/cleaned/'
    data = process_raw(data_path)
    data.to_csv('E:/nlp_models/test_data/SA_store_type/raw_data/raw_data.csv',index=False)
    # data = pd.read_csv('E:/nlp_models/test_data/SA_store_type/raw_data/raw_data.csv')
    check_raw = checker(data).check_duplicates
    tmp = cleaner(data)
    res = tmp.standard_process
    check_cleaned = checker(res).check_duplicates
    res.text = res.text_bk
    res = res.drop('text_bk',axis=1)
    res.to_csv('E:/nlp_models/test_data/SA_store_type/release/cleaned/cleaned_all.csv',index=False)
    assert res.isnull().sum().sum() == 0
    train, valid, test = split_data(res)
    train.to_csv(output_path+'train.csv')
    valid.to_csv(output_path+'valid.csv')
    test.to_csv(output_path+'test.csv')

