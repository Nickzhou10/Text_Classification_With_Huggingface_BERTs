# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 12:03:57 2023
@author: zhni2001
"""
import pandas as pd
import numpy as np
import re,time
from cleaning.data_chker import checker

class cleaner:
    '''    
        processing method:
            * standard_process: used for training; cleaning data using all processes
            * prediction_process: used for dfs to test; cleaning meaningless data found
            * deployment_process: for production process; cleaning NaNs and NaN-like data only.
            
        cleaning method:
            * _clean_NumOnly: cleaning rows with numbers only for selected classes
            * _clean_errors: cleaning names with 无中文, 无名
            * _clean_labels: cleaning texts with diff labels, drop rows with total<=3 & multi labels
            * _clean_lenthy: cleaning lengthy texts,control range by lo/uplimit and class by unimportant_classes
            * _clean_duplicated: cleaning duplicates by subset_for_dup
            *_clean_nan: cleaning nans as the final step
            
        __class__.__init__:
            * defines the parameters for the cleaning methods above
    '''

    def __init__(self):
        # for adding other info to text
        self.text_list = ['province','city','urban','text']
        # for cleaning errors
        self.skip_word = ['无名','无中文','无店名','提供'] 
        # for cleaning NumOnly
        self.unimportant_classes = ['tt','others'] 
        # for cleaning lenthy
        self.len_lolimit=1
        self.len_uplimit=30
        # for cleaning labels
        self.priority_list = ['hyper', 'super', 'cvs', 'mini', 'baby', 'tt', 'others'] 
        # for cleaning duplicates
        self.subset_for_dup = ['text']
    
    
    def checker(self, data):
        check = checker(data)
        self.duplicated = check.check_duplicates
        self.ck_by_bins = check.check_len_by_bins
        self.over_len_limit = check.over_len_limit

    
    def standard_process(self, data):
        res = data.copy()
        res = self.clean_data(res,self._clean_errors)
        res = self.clean_data(res,self._clean_format)
        res = self.clean_data(res,self._clean_lenthy)
        res = self.clean_data(res,self._clean_NumOnly)
        res = self.clean_data(res,self._clean_nan)
        res = self.clean_data(res,self._clean_labels)
        res = self.clean_data(res,self._clean_duplicated)
        return res
    
    
    def prediction_process(self, data):
        res = data.copy()
        res = self.clean_data(res,self._clean_errors)
        res = self.clean_data(res,self._clean_format)
        res = self.clean_data(res,self._clean_NumOnly)
        res = self.clean_data(res,self._clean_nan)
        return res
    
    
    def deployment_process(self, data):
        res = data.copy()
        res = self.clean_data(res,self._clean_errors)
        res = self.clean_data(res,self._clean_format)
        res = self.clean_data(res,self._clean_nan)
        return res
    
    
    @staticmethod
    def clean_data(df,clean_fuc):
        method_name = clean_fuc.__name__
        print(f"Applying {method_name} method...")
        start_time = time.time()
        cleaned_df = clean_fuc(df)
        time_used = round(time.time() - start_time,1)
        print(f"DF length before: {len(df)}; after {method_name} method: {len(cleaned_df)}")
        cleaned = len(df) - len(cleaned_df)
        print(f"...{method_name} method cleaned: {cleaned} rows; used {time_used} seconds")
        print(" ")
        return cleaned_df
    
    
    def _clean_errors(self,data):
        df = data.copy()
        df.text = df.text.astype(str)
        df.dropna(subset=['text'],inplace=True)
        df = df[~df['text'].str.contains('|'.join(self.skip_word))]
        return df
        
        
    def _clean_lenthy(self,data):
        df = data.copy()
        df['len']= df['text'].str.len()
        df = df[(df.len<self.len_uplimit)&(df.len>self.len_lolimit)]
        df = df.drop('len',axis=1)
        return df
        
    
    def _clean_NumOnly(self,data):
        df = data.copy()
        df = df[~(df['label'].isin(self.unimportant_classes) & df['text'].str.isdigit())]
        return df
    
    
    def _clean_labels(self,data):
        df = data.copy()
        df = self._merge_text(df)
        dup = checker(df).check_duplicates()
        ready_1 = df[df.text.isin(dup[dup.label_count==1].index)]
        not_ready = dup[dup.label_count!=1]
        not_ready_1 = not_ready[not_ready.total>3]
        not_ready_2 = not_ready_1.drop(['label_count','total'],axis=1)
        max_cols = []
        for index, row in not_ready_2.iterrows():
            max_val = row.max()
            max_cols_row = list(row[row == max_val].index)
            result = [x for x in max_cols_row if x in self.priority_list]
            result.sort(key=self.priority_list.index)
            final_result = result[0:1]
            max_cols.append(final_result[0])
            
        not_ready_2['label'] = max_cols   
        not_ready_3 = not_ready_2[['label']].reset_index()
        ready_2 = df.merge(not_ready_3,on=['text','label'],how='inner')
        df = pd.concat([ready_1,ready_2])
        df = self._split_text(df)
        return df

    
    def _merge_text(self,data):
        df = data.copy()
        df['text'] = df[self.text_list].apply(lambda x: '#'.join(x.astype(str)), axis=1)
        return df
        
    
    def _split_text(self,data):
        df = data.copy()
        df['text_bk'] = df['text'].apply(lambda x:x.split('#')[-1])
        return df
    
    
    def _clean_duplicated(self,data):
        df = data.copy()
        df = df.drop_duplicates(subset=self.subset_for_dup)
        return df
 

    def _clean_nan(self,data):
        df = data.copy()
        df = df.dropna(subset=['text'])
        return df
        
    
    def _clean_format(self,data):
        df = data.copy()
        pattern = r'(\d+店$|\s.+店$|NO+\d+$|NOS+\d+$|\({1}.+\){1}|\[{1}.+\]{1}|\{{1}.+\}{1}|（{1}.+）{1}|「{1}.+」{1})'
        df['text'] = df['text'].str.replace(pattern,'',regex=True)
        df['text'] = df['text'].apply(lambda x: re.sub(r'\(.*', '', x))
        df['text'] = df['text'].str.upper()
        df['text'] = df['text'].str.replace(' ', '')
        df['text'] = df['text'].str.replace('\)', '',regex=True)
        return df
        
      
if __name__ == '__main__':
    
    data = pd.read_csv('E:/nlp_models/test_data/SA_store_type/raw_data/raw_data.csv')
    res = cleaner().standard_process(data)
    #%% ck
    
    key = ''  # 无名 无中文
    test_old = res[res['text'].str.contains(key)==True]
    test_old['text'] = test_old['text'].apply(lambda x: re.sub(r'\(.*', '', x))
    data_ck = data[data.store_code.isin(test_old.store_code)]
    # test123 = res[res['text'].str.contains(key)==True]
    # ck = res.merge(data,how='right',on='store_code')
    # ck['is_equal'] = 0
    # ck.loc[ck['label_x'] == ck['label_y'], 'is_equal'] = 1
    # ck.is_equal.value_counts()
    #%% label test
    # test = checker(data).check_duplicates
    # ready = test[test.label_count==1]
    # ready_1 = data[data.text.isin(ready.index)]
    
    # not_ready = test[test.label_count!=1]
    # not_ready_1 = not_ready[not_ready.total>3]
    # not_ready_2 = not_ready_1.drop(['label_count','total'],axis=1)
    # max_cols = []
    # test_cols = []
    # for index, row in not_ready_2.iterrows():
    #     max_val = row.max()
    #     max_cols_row = list(row[row == max_val].index)
    #     priority_list = ['hyper', 'super', 'cvs', 'mini', 'baby', 'tt', 'others']
    #     result = [x for x in max_cols_row if x in priority_list]
    #     result.sort(key=priority_list.index)
    #     final_result = result[0:1]
    #     max_cols.append(final_result[0])
    #     test_cols.append(max_cols_row)
        
    # not_ready_2['label'] = max_cols   
    # not_ready_2['test_cols'] = test_cols 
    # not_ready_3 = not_ready_2[['label']].reset_index()
    # ready_2 = data.merge(not_ready_3,on=['text','label'],how='inner')
    # final_res = pd.concat([ready_1,ready_2])
    # dup = checker(final_res).check_duplicates
    
    
# def raw_data():
#     rename_map = {'fix_name':'text','shop_id':'store_code','source':'label',
#                   'nls_store_type':'label','adname_1':'province',
#                   'name_std[std_name]':'text','province_name':'province'}
#     columns = ['store_code', 'text', 'province', 'label']
#     data1 = pd.read_pickle('E:/nlp_models/sa_current/LSTM/2206/name2channel/data/202206_sa_train.pickle').rename(columns=rename_map)[columns]
#     data2 = pd.read_pickle('E:/nlp_models/sa_current/LSTM/2206/name2channel/data/202206_sa_train_black_list.pickle').rename(columns=rename_map)[columns]
#     data = pd.concat([data1,data2]).set_index(['store_code'])
#     return data