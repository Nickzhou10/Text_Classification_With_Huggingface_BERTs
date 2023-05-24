# -*- coding: utf-8 -*-

"""
Created on Mon May  8 10:37:31 2023

@author: zhni2001
"""

import pandas as pd
import numpy as np
import time
import torch
from imblearn.under_sampling import TomekLinks
from sklearn.feature_extraction.text import TfidfVectorizer
from tools.common_tools import label_to_id, generate_combinations
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split


class BalanceClasses:
    
    def __init__(self, data, random_state):
        self.data = data
        self.n_jobs = torch.get_num_threads()-2
        self.tokenizer = TfidfVectorizer()
        self.random_state = 42
        
    
    @staticmethod
    def balance_df(df,fuc):
        method_name = fuc.__name__
        print(f"Applying {method_name} method...")
        start_time = time.time()
        modified_df = fuc(df)
        time_used = round(time.time() - start_time,1)
        counts_before = df.label.value_counts()
        counts_after = modified_df.label.value_counts() 
        print(f'Before:\n {counts_before}') 
        print(f'After {method_name} method: \n {counts_after}')
        modified = len(modified_df) - len(df) 
        print(f"...{method_name} method changed: {modified} rows in total; used {time_used} seconds")
        print(" ")
        return modified_df
    
    
    def experiment1(self):
        df = self.data.copy()
        df = self.balance_df(df,self._random_undersampling)
        # df = self.balance_df(df,self._random_oversampling)
        train, valid = self.split_train_valid(df)
        return train, valid
        
    
    def _random_undersampling(self,data):
        df = data.copy()
        df, label_map = self._transform(df)
        sampling_strategy = self._auto_random_strategy(df)
        sampler = RandomUnderSampler(sampling_strategy=sampling_strategy,
                                          random_state=self.random_state)
        res = self._fit_transform(df, sampler, label_map)
        return res
    
    
    def _random_oversampling(self,data):
        df = data.copy()
        df, label_map = self._transform(df)
        sampler = RandomOverSampler(sampling_strategy='auto',
                                        random_state=self.random_state)
        res = self._fit_transform(df, sampler, label_map)
        return res
    
    
    def _geo_oversampling(self):
        pass
    
    
    
    def _fit_transform(self, df, sampler, label_map):
        text, label = sampler.fit_resample(df.text, df.label)
        text = self._transform_back(text)
        res = pd.DataFrame({'text':text,'label':label})
        res.text = res.text.apply(lambda x:''.join(map(str,x)))
        res.label = res.label.replace(label_map)
        return res
    
    
    def _transform_back(self,text):
        text = self.tokenizer.inverse_transform(text)
        return text
    
    
    def _transform(self,df):
        df.label, label_map = label_to_id(df.label)
        df.text = self.tokenizer.fit_transform(df.text)
        return df,label_map
    
    
    def _auto_random_strategy(self,df):
        stats = df.label.value_counts()
        top = stats.nlargest(len(stats)-round(len(stats)/2))
        max_count = top.iloc[0]
        second_max_count = top.iloc[1] # to be automated
        third_max_count = top.iloc[2]
        sampling_strategy = {cls: third_max_count if count == max_count or
                             count == second_max_count else count 
                             for cls, count in stats.items()}
        return sampling_strategy


    def split_train_valid(self,df):
        train, valid = train_test_split(df, test_size=0.125, 
                                        random_state=self.random_state, 
                                        stratify=df.label)
        return train, valid


if __name__ == '__main__':
    
    data1 = pd.read_csv('data/train.csv',index_col=0)
    data2 = pd.read_csv('data/valid.csv',index_col=0)
    data = pd.concat([data1,data2])
    Balance = BalanceClasses(data)
    train, valid = Balance.experiment1()