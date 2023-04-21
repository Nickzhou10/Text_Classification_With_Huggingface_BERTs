# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 11:39:56 2023

@author: zhni2001
"""

import pandas as pd
import numpy as np
import re,time

class initial_cleaning:
    
    def __init__(self,data_path):
        self.data_path = data_path
        
        
    def data_loader(self,df):
        return (df.
                rename(columns={'shop_id':'store_code',
                                'fix_name':'store_name',
                                'std_name':'store_name',
                                'nls_store_type':'store_type',
                                'adname_1':'province_name'}).
                set_index('store_code').
                filter(['store_name','store_type','province_name']).
                dropna())
        
    
    @property
    def model_train_data(self):
        store_train = self.data_loader(
            pd.read_csv(self.data_path+'store_train.csv', encoding='utf-8'))
        others = self.data_loader(pd.read_excel(self.data_path+'store_others.xlsx').
                           assign(store_type='OTHERS'))
        data = pd.concat([store_train, others])
        data = (data.rename(columns={'store_name':'text',
                                     'store_type':'label',
                                     'province_name':'province'}).
                     astype('string'))
        return data.reset_index()
    
    

            
            
            
            