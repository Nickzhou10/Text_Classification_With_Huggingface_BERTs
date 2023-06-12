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
from tools.similarity import Similarity
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from itertools import product


class BalanceClasses:
    
    def __init__(self, train, valid, random_state = 42, n_jobs = 10, split_ratio=0.125):
        self.train = train
        self.valid = valid
        self.n_jobs = n_jobs
        self.split_ratio = split_ratio
        self.tokenizer = TfidfVectorizer()
        self.random_state = random_state
        self.paralell = True
        self.n_similar = Similarity(n_jobs=self.n_jobs,
                                    drop_threshold=0.75,
                                    by='city')
    

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
        df = self.balance_df(df,self._similarity_undersampling)
        df.to_csv('data/data_b.csv')
        df = self.balance_df(df,self._geo_oversampling)
        train, valid = self.split_train_valid(df)
        return train, valid 
        
    
    def experiment2(self): 
        train, valid = self.train.copy(), self.valid.copy()
        # train = self.balance_df(train,self._random_undersampling)
        train = self.balance_df(train,self._geo_oversampling)
        return train, valid 
    
    
    def _similarity_undersampling(self, data):
        df = data.copy()
        stats = df.label.value_counts()
        top = stats.nlargest(len(stats)-round(len(stats)/2)-1)
        threshold = top.iloc[-1]
        res = []
        for classes in stats.index:
            df_sub = df[df['label'].isin([classes])]
            if len(df_sub) >= threshold:
                if self.paralell: 
                    df_sub = self.n_similar.parallelize_run(df_sub)
                else:
                    df_sub = self.n_similar.similarity_filter(df_sub)
                res.append(df_sub)
            else:
                res.append(df_sub)
        res = pd.concat(res, axis=0).drop_duplicates()
        return res
    
    
    def _geo_oversampling(self, data):
        df = data.copy()
        stats = df.label.value_counts()
        top = stats.nlargest(len(stats)-round(len(stats)/2)+1)
        low = stats[~stats.index.isin(top.index)].index.tolist()
        threshold = top.iloc[-1]
        res = []
        for label in low:
            df_sub = df[df['label']==label]
            combinations = pd.merge(df_sub[['text','label']], 
                                    df_sub[['province', 'city', 'urban']], 
                                    how='cross')
            if len(combinations) >= threshold:
                combinations = combinations.sample(threshold)
                res.append(combinations)
            else:
                res.append(combinations)
        modified_df = pd.concat([df,pd.concat(res, axis=0)], axis=0)
        return modified_df
    
      
    def _random_undersampling(self, data):
        df = data.copy()
        stats = df.label.value_counts()
        top = stats.nlargest(len(stats)-round(len(stats)/2)-1)
        low = stats[~stats.index.isin(top.index)].index.tolist()
        threshold = top.iloc[-1]
        res = []
        for classes in stats.index:
            df_sub = df[df['label'].isin([classes])]
            if len(df_sub) > threshold:
                df_sub = df_sub.sample(threshold)
                res.append(df_sub)
            else:
                res.append(df_sub)
        res = pd.concat(res, axis=0).drop_duplicates()
        return res

    
    def _random_oversampling(self,data):
        df = data.copy()
        df, label_map = self._transform(df)
        sampler = RandomOverSampler(sampling_strategy='auto',
                                        random_state=self.random_state)
        res = self._fit_transform(df, sampler, label_map)
        return res
    
    
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
   
    
    def _random_undersampling_b(self,data):
        df = data.copy()
        df, label_map = self._transform(df)
        sampling_strategy = self._auto_random_strategy(df)
        sampler = RandomUnderSampler(sampling_strategy=sampling_strategy,
                                          random_state=self.random_state)
        res = self._fit_transform(df, sampler, label_map)
        return res
    
    
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
        train, valid = train_test_split(df, test_size=self.split_ratio, 
                                        random_state=self.random_state, 
                                        stratify=df.label)
        return train, valid

#%%
if __name__ == '__main__':
    df = pd.concat([pd.read_csv('../niq_sa_pre/data/valid.csv',index_col=0),
                      pd.read_csv('../niq_sa_pre/data/train.csv',index_col=0)],
                      axis=0).drop_duplicates()
    Balance = BalanceClasses(df)
    train = Balance.experiment2()
    
    #%%
    # df = pd.concat([pd.read_csv('../niq_sa_pre/data/valid.csv',index_col=0),
    #                   pd.read_csv('../niq_sa_pre/data/train.csv',index_col=0)],
    #                   axis=0).drop_duplicates()
    # df = df.set_index(['store_code','source'])
    # stats = df.label.value_counts()
    
    # top = stats.nlargest(len(stats)-round(len(stats)/2)+1)
    # low = stats[~stats.index.isin(top.index)].index.tolist()
    # threshold = top.iloc[-1]
    # res = []
    # df_sub = df[df.label=='hyper']
    # combinations = pd.merge(df_sub[['text','label']], 
    #                         df_sub[['province', 'city', 'urban']], 
    #                         how='cross')
    # if len(combinations) >= threshold:
    #     combinations = combinations.sample(threshold)
    #     res.append(combinations)
    # else:
    #     res.append(combinations)
    # test = pd.concat(res, axis=0).reset_index(drop=True)
    # modified_df = pd.concat([df,test], axis=0).reset_index(drop=True).drop_duplicates()
    # # test = .drop_duplicates()
    # print(modified_df.label.value_counts())
    # test = test[test.text=='沃尔玛']
    