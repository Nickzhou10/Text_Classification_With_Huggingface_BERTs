# -*- coding: utf-8 -*-
"""
Created on Sun May 28 02:11:46 2023

@author: nick
"""

import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')


class Similarity:
    
    def __init__(self, n_jobs=1, drop_threshold=0.8, by='province'):
        self.n_jobs = n_jobs
        self.drop_threshold = drop_threshold
        self.by = by
    
    
    def parallelize_run(self, data):
        df = data.copy()
        # load df to a dict grouped by predefined 'by'
        grouped_dict = {key: group.reset_index(drop=True) for key, group in df.groupby(self.by)}
        # parallelize
        results = Parallel(n_jobs=self.n_jobs)(delayed(self.similarity_filter)(data)
                                      for data in grouped_dict.values())
        merged_df = pd.concat(results).reset_index()
        return merged_df


    def similarity_filter(self, data):
        df = data.copy()
        index_to_drop = set()
        for index, row in df.iterrows():
            if index in index_to_drop:
                continue
            text1 = row['text']
            res = self._similarity_compare(text1,df[df.index != index])
            index_to_drop.update(res[res.scores>=self.drop_threshold].index)
        
        df = df[~df.index.isin(list(index_to_drop))] 
        return df


    def _similarity_compare(self, text1, df)  :
        '''
        input:
            text1: str
            df: dataframe
        output:
            df: dataframe with scores
        '''
        df['scores'] = df['text'].apply(lambda x:self._similarity_score(text1, x))
        return df


    def _similarity_score(self, text1, text2):
        '''
        input:
            text1: str
            text2: str
        output:
            scores: float
        '''
        scores = SequenceMatcher(None, text1, text2).ratio()
        return scores 


if __name__ == '__main__':
    # unit test
    df = pd.read_csv('E:/nlp_models/sa_bert_deployment/niq/nlp_sa_test/data/valid.csv',index_col=0)
    df = df[df.label.isin(['tt'])].head(400)
    similarity = Similarity(n_jobs=22, drop_threshold=0.6)
    res = similarity.similarity_filter(df)
    para_res = similarity.parallelize_run(df)
