# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 12:01:57 2023

@author: zhni2001
"""
import time,os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from train import TrainModel
from tools.common_tools import label_to_id, add_info, class_report
from tools.preprocessing import Preprocess
from tools.balance import BalanceClasses
from cleaning.data_cleaner import cleaner
from model import AutoModelForClassification
import warnings
warnings.filterwarnings("ignore")


class TestModel:
    
    def __init__(self, model_name, state_dict_path, pretrained_path, n_jobs=32):
        
        self.model_name = model_name
        self.state_dict_path = state_dict_path        
        self.pretrained_path = pretrained_path
        self.n_jobs = n_jobs
        torch.set_num_threads(self.n_jobs)
        self.batch_size = 64
        self.output_cols = ['shop_id','sa_id','source_type','text']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_map = {0: 'baby', 1: 'cvs', 2: 'hyper', 3: 'mini', 
                          4: 'others', 5: 'super', 6: 'tt'}
        self.inverse_label = dict((v,k) for k, v in self.label_map.items())
        # load df to tensor processor
        self.processor = Preprocess(self.model_name, batch_size=self.batch_size,
                                    cache_dir=self.pretrained_path, n_jobs=self.n_jobs)
        # load model
        self.model = AutoModelForClassification(self.model_name, 
                                                self.pretrained_path).to(self.device)
        # load trained parameters
        self.model.load_state_dict(torch.load(self.state_dict_path,
                                   map_location=self.device))
        print('using: ' + str(torch.get_num_threads()) + ' cores')
        
        
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
    
    
    def adjust_df(self, df):
        mapping = {'std_name':'text','adname_1':'province',
                   'adname_2':'city','urban_tag':'urban'}
        df = df.reset_index().rename(columns=mapping)
        return df
    
    
    def deployment(self, pre):
        pre_raw = self.adjust_df(pre)
        pre_ready = cleaner(pre_raw).deployment_process()
        pre_ready = add_info(pre_ready)
        pre_loader = self.processor.process_data(pre_ready, deployment=True)
        start_time = time.time()
        predicted, info_list = self.deployment_predict(pre_loader)
        time_used = round(time.time() - start_time,1)
        print(f"used {time_used} seconds")
        predicted.index = info_list
        predicted = pd.merge(predicted, 
                             pre_ready[self.output_cols], 
                             left_index=True, right_index=True, 
                             how='outer')
        predicted['mt'] = predicted['cvs'] + predicted['hyper'] + predicted['mini']+ predicted['super']
        predicted = predicted.set_index(self.output_cols).add_prefix('pred_prob_')
        predicted = predicted.fillna(0).round(6)
        return predicted
        
    
    def deployment_predict(self,test_loader):
        self.model.eval()
        predicted = []
        info_list = []
        with torch.no_grad():
            for batch_idx, (input_ids, attention_mask, info) in enumerate(test_loader):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                outputs = torch.softmax(outputs, dim=1)
                predicted.extend(outputs.cpu().numpy())
                info_list.extend(info.cpu().numpy())
        df = pd.DataFrame(predicted, columns=range(len(predicted[0])))
        df = df.rename(columns=self.label_map)
        return df, info_list

if __name__ == '__main__': 
    # unit test
    model_name = 'bert-base-chinese'  
    state_dict_path = '../nlp_sa_test/output/current_best_balanced/model.pth'
    pretrained_path = '../nlp_sa_test/pretrained/bert-base-chinese'
    # load test data
    test_df = pd.read_csv('../nlp_sa_test/data/test.csv', index_col=0)
    tmp = TestModel(model_name, state_dict_path, pretrained_path)
    # tmp.test(test_df)
    
    # prediction
    pre = pd.read_csv('../niq_sa_pre/data/pre.csv',index_col=0).sample(100)
    res = tmp.deployment(pre)
    
    
    
    
      
    
    