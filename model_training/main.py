# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 16:14:54 2023

@author: nick
"""
import time,os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tools.common_tools import label_to_id, add_info
from preprocessing import Preprocess
from train import TrainModel
from balance import BalanceClasses
# from model import AutoModelForClassification
from model_test import AutoModelForClassification
import warnings
warnings.filterwarnings("ignore")


class TestModels:
    def __init__(self,model_name,experiment_name,data_path):
        
        # define paths
        self.experiment_name = experiment_name
        self.data_path = data_path
        self.output_path = 'output/'+ self.experiment_name
        self.model_name = model_name
        self.model_download_path = 'pretrained/' + model_name
        self.report_save_path = self.output_path 
        self.model_save_path = self.report_save_path + 'model.pth'
        os.makedirs(self.model_download_path,exist_ok=True)
        os.makedirs(self.report_save_path,exist_ok=True)
        # load data
        self.train = add_info(pd.read_csv(self.data_path+'train.csv', index_col=0))
        self.valid = add_info(pd.read_csv(self.data_path+'valid.csv', index_col=0))
        self.test = add_info(pd.read_csv(self.data_path+'test.csv', index_col=0))
        # model config
        self.random_state = 42
        self.batch_size = 512
        self.max_epochs = 10
        self.learning_rate = 1e-4
        self.num_classes = len(set(self.train.label))
        self.finetune_all = True
        # define tensor data loading processor
        self.processor = Preprocess(self.model_name,batch_size=self.batch_size,
                                    cache_dir=self.model_download_path)
         
        
    def balance(self):
        print('Solving Class Imbalance...')
        data = pd.concat([self.train,self.valid])
        Balance = BalanceClasses(data, self.random_state)
        self.train,self.valid = Balance.experiment1()
        
        
    def training(self):
        print('Starting loading model data to device...')
        # load data and encode label
        train, valid = self.train.copy(), self.valid.copy()
        valid.label, label_map = label_to_id(valid.label)
        train.label, label_map = label_to_id(train.label)
        
        # data to tensor
        train_loader = self.processor.standard_process(train)
        valid_loader = self.processor.standard_process(valid)
        
        # define model
        print('Loading pre-trained model...')
        model = AutoModelForClassification(self.num_classes, 
                                           self.model_name, 
                                           cache_dir=self.model_download_path, 
                                           finetune_all=self.finetune_all)
        
        # define trainer
        print('Starting training model...')
        self.trainer = TrainModel(model, train_loader, valid_loader, label_map,
                                  self.batch_size, self.num_classes,
                                  self.max_epochs, self.learning_rate,
                                  self.random_state)
        
        # training & testing
        self.trainer.train_model()
        self.trainer.save_model(self.model_save_path)


    def run_test(self):
        # clean GPU memory
        torch.cuda.empty_cache()
        test = self.test.copy()
        test.label, label_map = label_to_id(test.label)
        test_loader = self.processor.standard_process(test)
        self.trainer.test(test_loader,self.model_save_path)


    def save_report(self):
        os.makedirs(self.report_save_path,exist_ok=True)
        writer = pd.ExcelWriter(self.report_save_path + 'output.xlsx',
                                engine='xlsxwriter')
        parameters = self._log_parameter()
        parameters.to_excel(writer,sheet_name='parameter_report')
        self.trainer.test_report.to_excel(writer,sheet_name='test_report')
        self.trainer.train_report.to_excel(writer,sheet_name='train_report')
        self.trainer.valid_report.to_excel(writer,sheet_name='valid_report')
        writer.close()
        
        
    def _log_parameter(self):
        parameters = {
            'Experiment Name':self.experiment_name,
            'Pretrained Model Name':self.model_name,
            'seed':self.random_state,
            'finetune_all':self.finetune_all,
            'Max_Epochs': self.trainer.max_epochs,
            'Epochs Used': self.trainer.epochs_used,
            'Early Stopping Patience': self.trainer.patience,
            'Total Seconds Used': self.trainer.total_time_used,
            'Learning Rate': self.trainer.lr,
            'Batch Size': self.trainer.batch_size,
            'Warmup': self.trainer.warmup,
            'Warmup Method': self.trainer.scheduler_method,
            'No. Warmup Steps': self.trainer.num_warmup_steps,
            'No. Training Steps': self.trainer.num_training_steps,
            'Warmup': self.trainer.warmup,
                     }
        df = pd.DataFrame(parameters, index=['Parameters']).T
        df.Parameters = df.Parameters.astype(str)
        return df
        
        
if __name__ == '__main__': 
    # clean GPU memory
    torch.cuda.empty_cache()
    torch.cuda.memory_summary()
    # testing
    experiment_name = 'current_best2/'
    model_name = 'bert-base-chinese'  
    # model_name = 'Davlan/distilbert-base-multilingual-cased-ner-hrl'
    data_path = 'data/'
    test = TestModels(model_name,experiment_name,data_path)
    test.balance()
    test.training()
    test.run_test() 
    test.save_report()
    
    
    
    
    
    
    