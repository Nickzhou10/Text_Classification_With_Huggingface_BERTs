# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 16:14:23 2023

@author: nick
"""

import time,os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from train import TrainModel
from tools.common_tools import label_to_id, add_info, class_report, adjust_df
from tools.preprocessing import Preprocess
from tools.balance import BalanceClasses
from cleaning.data_cleaner import cleaner
from model import AutoModelForClassification
import warnings
warnings.filterwarnings("ignore")


class ReleasePrediction:
    
    def __init__(self, config):
        self.output_cols = ['shop_id','sa_id','source_type','text']
        self.output_cols_unit = ['source_type','text']
        self.processor = Preprocess(self.model_name, self.pretrained_path,
                                    batch_size=self.batch_size,
                                    n_jobs=self.n_jobs)
        # load model
        self.model = AutoModelForClassification(self.model_name, 
                                                self.pretrained_path).to(self.device)
        # load trained parameters
        self.model.load_state_dict(torch.load(self.state_dict_path,
                                   map_location=self.device))
        print('using: ' + str(torch.get_num_threads()) + ' cores')
        print('using: ' + str(self.device))
        
              
    def test(self, test_df):
        test_df = add_info(test_df)
        test_df.label = test_df.label.replace(self.inverse_label)
        test_loader = self.processor.standard_process(test_df)
        start_time = time.time()
        print('starting predicting...')
        predicted = self.predict_test(test_loader)
        time_used = round(time.time() - start_time,1)
        print(f"used {time_used} seconds")
        return predicted
        
    
    def predict_test(self, test_loader):
        self.model.eval()
        predicted = []
        true_labels = []
        with torch.no_grad():
            for batch_idx, (input_ids, attention_mask, labels) in enumerate(test_loader):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                outputs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                predicted.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        self.test_report = class_report(true_labels, predicted, self.label_map)
        print(self.test_report) 
        return self.test_report
    
    
    def deployment(self, pre):
        pre_raw = adjust_df(pre)
        pre_ready = cleaner().deployment_process(pre_raw)
        pre_ready = add_info(pre_ready)
        print('starting process data to tensor...')
        pre_loader = self.processor.process_data(pre_ready, deployment=True)
        start_time = time.time()
        print('starting predicting...') 
        predicted, info_list = self.deployment_predict(pre_loader)
        time_used = round(time.time() - start_time,1)
        print(f"used {time_used} seconds")
        predicted.index = info_list
        predicted = pd.merge(predicted, 
                             pre_ready[self.output_cols], 
                             left_index=True, right_index=True, 
                             how='outer')
        predicted = predicted.fillna(0)
        predicted['mt'] = predicted['cvs'] + predicted['hyper'] + predicted['mini']+ predicted['super']
        predicted = predicted.set_index(self.output_cols).add_prefix('pred_prob_')
        predicted = predicted.fillna(0).round(6)
        return predicted
        
    
    def deployment_predict(self, pre_loader):
        self.model.eval()
        predicted = []
        info_list = []
        with torch.no_grad():
            for batch_idx, (input_ids, attention_mask, info) in enumerate(pre_loader):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                outputs = torch.softmax(outputs, dim=1)
                predicted.extend(outputs.cpu().numpy())
                info_list.extend(info.cpu().numpy())
        df = pd.DataFrame(predicted, columns=range(len(predicted[0])))
        df = df.rename(columns=self.label_map)
        return df, info_list


    def unit_pre(self, pre):
        print('starting process data to tensor...')
        pre_loader = self.processor.process_data(pre, deployment=True)
        start_time = time.time()
        print('starting predicting...') 
        predicted, info_list = self.deployment_predict(pre_loader)
        time_used = round(time.time() - start_time,1)
        print(f"used {time_used} seconds")
        predicted.index = info_list
        predicted = pd.merge(predicted, 
                             pre[['text']], 
                             left_index=True, right_index=True, 
                             how='outer')
        
        return predicted

if __name__ == '__main__': 
    
    pre = pd.read_csv('data/sa_202303_raw.csv',index_col=0)
    pre.loc[pre['adname_1'].str.contains('新疆')==True,['adname_1','adname_2']] = ''
    pre.loc[pre['adname_1'].str.contains('西藏')==True,['adname_1','adname_2']] = ''
    
    
    pretrained_path = '../nlp_sa_test/pretrained/bert-base-chinese'
    final_model_path = '../nlp_sa_test/output/bert-base-chinese_refit_test/model.pth'
    final = TestModel(model_name, final_model_path, pretrained_path, n_jobs=n_jobs)
       
    size = 1048576
    df_parts = []
    df = pre.copy()
    for i in range(0, len(df), size):
        df_part = df[i:i+size]
        df_part = final.deployment(df_part)
        print("Batch", i//size + 1)
        df_parts.append(df_part)
    res = pd.concat(df_parts)
    ck = res.reset_index()
    final = release(pre, res)
    final.to_csv('sa_202303_release.csv')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    