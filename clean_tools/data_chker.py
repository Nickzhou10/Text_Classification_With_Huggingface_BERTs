# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 11:59:26 2023

@author: zhni2001

1. 保证相同text仅对应一种label 如果有多label也必须有其他信息辅助区分 如地理位置
2. 保证train test绝对独立比如去重，以防data leakage造成的测试结果很好，生产中很差
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re,time

class checker:
    '''
    clean text df
    
    check method:
        * check_classes: checking No. classes for each label
        * check_duplicates: checking duplicates by labels
        * check_len: checking length for texts

    '''
    
    def __init__(self,data):
        self.data = data
        self.len_raw = len(data)
        self.len_limit = 30
        
    @property
    def check_classes(self):
        print(self.data.
              groupby('label').
              count()['text'].reset_index().
              sort_values(by='text',ascending=False))
    
    
    @property
    def check_duplicates(self):
        df = self.data.copy()
        text_label_count = df.groupby(['text', 'label']).size().reset_index(name='count')
        pivot_tmp = text_label_count.pivot_table(index='text', columns='label', values='count', fill_value=0)
        pivot = pivot_tmp.reset_index().rename_axis(None, axis=1)
        pivot = pivot.set_index('text')
        pivot['label_count'] = (pivot>0).sum(axis=1)
        pivot['total'] = pivot.drop(['label_count'],axis=1).sum(axis=1)
        return pivot


    @property
    def check_len_by_bins(self):
        text_len = self.data.copy()
        text_len['len']= text_len['text'].str.len()
        self.ck_len_all = text_len.copy()
        self.over_len_limit = text_len[text_len.len>self.len_limit]
        text_len.len.hist(bins=100)
        res = (text_len.
               filter(['label','len'],axis=1).
               groupby(['label', 
                        pd.cut(text_len.len, 
                                [0,10,20,30,40,50,60,100])]).
               size().
               unstack())
        return res
    
        
     #%%       
if __name__ == '__main__':
    
    data = pd.read_csv('E:/nlp_models/test_data/SA_store_type/raw_data/latest_data.csv')
    tmp = checker(data)
    test = tmp.check_duplicates
    test1 = tmp.check_len_by_bins
    len_all = tmp.ck_len_all
    len_1 = len_all[len_all.len==1]
    ck_over_30 = tmp.over_len_limit
    #%% 
    # numeric_rows = data['text'].str.match(r'^\d+$')
    # ck_char = data[numeric_rows]
    # key = '+' 
    # test_word = data[data['text'].str.contains(key)==True]
    # tmp = test_word[test_word.label=='cvs']
    # ck1 = ck.loc[ck.text=='cvs']
    #%%
    ready = test[test.label_count==1].index
    not_ready = test[test.label_count!=1]
    not_ready_1 = not_ready
    ck = data[data.text.isin(ready)]
    # test =  
        
    df = data.dropna()
    ck = df[df['text'].str.match(r'^\d+$')]
    
        
        
        
        
        
        
        
        
        
        
        
        
        