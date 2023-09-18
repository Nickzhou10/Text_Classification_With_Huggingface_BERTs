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
from config.train_config import TrainConfig
from config.incremental_train_config import IncreTrainConfig
from config.predication_config import PredictionConfig
from transformers import pipeline
import warnings 
warnings.filterwarnings("ignore")

class TrainClassificationModel:
    """ 
    It is a client code to processes df to tensor df automatically; download 
    pretrained model from Huggingface with automatic selector of model type and
    its corresponding tokenizers; automatically fine tuning and train on your
    own data with customizable parameters. You can try different models like:
        - model_name = 'uer/roberta-base-finetuned-chinanews-chinese'
        - model_name = 'tuhailong/cross_encoder_roberta-wwm-ext_v1'
        etc, and etc...
        
    All you need is just to input the name of the pretrained model 
    you wish to use, the train&valid datasets, and the code will 
    do the rest for you to generate a trained model and classfication reports.
    
    For incremental training, simply use the incremental_train_config.py and keep 
    incremental_training = True and change the other training settings accordingly
    
    Shape:
        - dataset: df with df.label and df.text is a must for training;
                   df with df.text is a must for prediction.
    
    Attributes:
        - must_col: defines the must-to-have columns for the train/test dataframe
    Methods:
        - train(df, train_config, required): input train df and train config to 
        get the trained model state dict in the pre-defined output path.
        While training, classfication reports will be generated once an epoch is finished. 
        - test(df, required): input test set to get the test report printed
        - predict_text(train_config, pre_config, text, required): predict a str/list of text


    """
    
    def __init__(self):
        self.must_col = ['text', 'label']
        
    
    def train(self, df, config):
        self.config = config 
        os.makedirs(self.config.output_path, exist_ok=True)
        os.makedirs(self.config.pretrained_cache_path, exist_ok=True)
        processor = Preprocess(self.config)
        train_loader, valid_loader = self._train_preprocess(df, processor)
        # initialize the model
        model = AutoModelForClassification(self.config)
        # initialize the trainer
        trainer = TrainModel(model, self.config)
        # training and save to output path
        trainer.train_model(train_loader, valid_loader)
        trainer.save_model(self.config.model_output_path)
        
        
    def test(self, test_df, config):
        assert test_df.columns.isin(self.must_col).sum() == len(self.must_col), \
            r'make sure that df contains both columns text and label'
        label_map = pd.read_pickle(config.output_path + '/label_map.pkl')
        test_df.label = test_df.label.replace(label_map['inverse_mapping'])
        config.mapping_dict['mapping'] = label_map['mapping']
        processor = Preprocess(config)
        test_loader = processor.process_data(test_df)
        model = AutoModelForClassification(config)
        model.load_state_dict(torch.load(config.model_output_path,
                                         map_location=config.device))
        trainer = TrainModel(model, config)
        _, self.test_report = trainer.evaluate(test_loader)
        print(self.test_report)

    
    def predict_text(self, train_config, pre_config, text):
        model = AutoModelForClassification(train_config)
        model.load_state_dict(torch.load(pre_config.model_state_dict_path,
                                         map_location=pre_config.device))
        processor = Preprocess(train_config)
        pred_loader = processor.process_pred(text)
        label_map = pd.read_pickle(pre_config.label_map_path)['mapping']
        df = TrainModel(model, train_config).predict(model, pred_loader)
        df = df.rename(columns=label_map).round(4)
        df.index = text
        return df
    
    
    def _train_preprocess(self, df, processor):
        assert  df.columns.isin(self.must_col).sum() == len(self.must_col), \
            r'make sure that df contains both columns text and label'
        assert len(set(df.label))==self.config.num_classes
        # generate and apply mapping for labels and save as json
        if self.config.incremental_training:
            self.config.mapping_dict = pd.read_pickle(self.config.label_path)
        else:
            self.config.mapping_dict = label_map(df.label)
        with open(self.config.output_path + '/label_map.pkl', 'wb') as f:
            pickle.dump(self.config.mapping_dict, f)
        df.label = df.label.replace(self.config.mapping_dict['inverse_mapping'])
        # train valid split, controled by split ratio
        train_df, valid_df = split_df(df, self.config.vali_split_ratio,
                                      self.config.random_state)
        # improve class imbalance for training set if necessary
        if self.config.balance_required: #!!! customization required
            Balance = BalanceClasses(train_df, self.config.random_state)
            train_df = Balance.experiment2()
        # to dataloader
        train_loader = processor.process_data(train_df)
        valid_loader = processor.process_data(valid_df)
        return train_loader, valid_loader
    
    
if __name__ == '__main__':

    # train all data
    train_df = pd.read_csv('data/train_df.csv', index_col=0)
    test_df = pd.read_csv('data/test.csv', index_col=0)
    Main = TrainClassificationModel()
    Main.train(train_df, TrainConfig)
    Main.test(test_df, TrainConfig)
    # train incremental data
    incre_df = pd.read_csv('data/incre_df.csv', index_col=0)
    IncreMain = TrainClassificationModel()
    IncreMain.train(incre_df, IncreTrainConfig)
    IncreMain.test(test_df, IncreTrainConfig)
    
    #%%
    # prediction test
    text = ['巴拉巴拉','SEVEN ELEVEN']
    res = Main.predict_text(TrainConfig, PredictionConfig, text)




