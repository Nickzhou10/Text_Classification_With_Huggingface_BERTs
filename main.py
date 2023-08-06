# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 16:14:54 2023

@author: nick
"""
import time,os
import pandas as pd
import numpy as np
import torch
import pickle
import torch.nn as nn
from tools.common_tools import label_to_id, add_info, split_df, adjust_df, label_map
from tools.preprocessing import Preprocess
from tools.balance import BalanceClasses
from model import AutoModelForClassification
from train import TrainModel
from test_model import TestModel
from config.train_config import TrainConfig

import warnings 
warnings.filterwarnings("ignore")

class TrainClassificationModel:
    """ 
    It is a client code to processes df to tensor df automatically; download 
    pretrained model from Huggingface with automatic selector of model type and
    its corresponding tokenizers; automatically fine tuning and train on your
    own data. You can try different models like:
        - model_name = 'uer/roberta-base-finetuned-chinanews-chinese'
        - model_name = 'tuhailong/cross_encoder_roberta-wwm-ext_v1'
        etc, and etc...
        
    All you need is just to input the name of the pretrained model 
    you wish to use, the train&valid datasets, and the code will 
    do the rest for you to generate a trained model and classfication reports.

    Shape:
        - dataset: df with df.label and df.text is a must for training;
                   df with df.text is a must for prediction.

    Args:
        model_name (str, required): model name copied from huggingface 
        experiment_name (str, required): place to save the pre-trained model
        n_jobs (int, optional): cpu cores to use for DataLoader & CPU training(if no GPU avaliable).
            Defaults to 8.
            
    Methods:
        - balance(): ways to perform both undersampling and oversampling
        - training(): input train&valid sets to get a trained model in output path
        auto-generating reports once an epoch is finished. 
        - testing(): input test set to get the test report printed
        - save_report(): save the train, valid, test reports and parameters used to a xlsx file
        
    """
    
    def __init__(self, config):
        self.config = config        
        self.col = ['text', 'label']
        os.makedirs(self.config.output_path, exist_ok=True)
        os.makedirs(self.config.pretrained_cache_path, exist_ok=True)
        
    
    def _train_preprocess(self, df, processor):
        assert  df.columns.isin(self.col).sum() == len(self.col), \
            r'make sure that df contains both columns text and label'
        # generate and apply mapping for labels and save as json
        self.config.mapping_dict = label_map(df.label)
        self.label_map_path = self.config.output_path + '/label_map.pkl'
        with open(self.label_map_path, 'wb') as f:
            pickle.dump(self.config.mapping_dict, f)
        df.label = df.label.replace(self.config.mapping_dict['inverse_mapping'])
        # train valid split, controled by split ratio
        train_df, valid_df = split_df(df, self.config.vali_split_ratio,
                                      self.config.random_state)
        # improve class imbalance for training set if necessary
        if self.config.balance_required: #!!!
            Balance = BalanceClasses(train_df, self.config.random_state)
            train_df = Balance.experiment2()
        # to dataloader
        train_loader = processor.process_data(train_df)
        valid_loader = processor.process_data(valid_df)
        return train_loader, valid_loader

        
    def train(self, df):
        processor = Preprocess(self.config)
        train_loader, valid_loader = self._train_preprocess(df, processor)
        # initialize the model
        model = AutoModelForClassification(self.config)
        # initialize the trainer
        trainer = TrainModel(model, self.config)
        # training and save to output path
        trainer.train_model(train_loader, valid_loader)
        trainer.save_model(self.config.model_output_path)
        
        
    def test(self, test_df):
        assert  test_df.columns.isin(self.col).sum() == len(self.col), \
            r'make sure that df contains both columns text and label'
        label_map = pd.read_pickle(self.config.output_path + '/label_map.pkl')
        test_df.label = test_df.label.replace(label_map['inverse_mapping']) 
        processor = Preprocess(self.config)
        test_loader = processor.process_data(test_df)
        model = AutoModelForClassification(self.config)
        trainer = TrainModel(model, self.config)
        _, self.test_report = trainer.evaluate(test_loader)
        print(self.test_report)
        
    
    def incremental_train(self, incre_df, trained_model_path):
        assert self.config.incremental_training == True, ''
        

    def predict_df(self, model_path, df):
        pass
    
    
    def predict_str(self, model_path, string):
        pass
    
    def _predict_preprocess(self, df, processor):
        pass
    
    
if __name__ == '__main__':
    train_df = pd.read_csv('data/train_df.csv', index_col=0).sample(20000)
    test_df = pd.read_csv('data/test.csv', index_col=0).sample(4000)
    # pre_df = pd.read_csv('data/pre.csv', index_col=0).sample(10000)
    Main = TrainClassificationModel(TrainConfig)
    Main.train(train_df)
    Main.test(test_df)
    
    














