# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 16:14:54 2023

@author: nick
"""
import time,os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tools.common_tools import label_to_id, add_info, split_df, release
from tools.preprocessing import Preprocess
from tools.balance import BalanceClasses
from model import AutoModelForClassification
from train import TrainModel
from test_model import TestModel
import warnings
warnings.filterwarnings("ignore")
 

class BertTrainer:
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
        
    !!! TO BE UPDATED:
        1. AUTO LABEL ENCODING & AUTO ENCODE BACK
        before updating you may need to adjest the label map&No.class mannually
        (e.g., self.label_map)
    """
    
    def __init__(self,model_name,experiment_name, n_jobs=8):
        # define paths
        self.experiment_name = experiment_name
        self.output_path = 'output/'+ self.experiment_name
        self.model_name = model_name
        self.model_download_path = 'pretrained/' + model_name
        self.report_save_path = self.output_path 
        self.model_save_path = self.report_save_path + 'model.pth'  
        os.makedirs(self.model_download_path,exist_ok=True)
        os.makedirs(self.report_save_path,exist_ok=True)
        # model config 
        self.n_jobs = n_jobs
        self.random_state = 42
        self.batch_size = 256  
        self.max_epochs = 2
        self.learning_rate = 1e-5
        self.finetune_all = True
        self.balance_required = False
        torch.set_num_threads(self.n_jobs)
        self.label_map = {0: 'baby', 1: 'cvs', 2: 'hyper', 3: 'mini', 
                          4: 'others', 5: 'super', 6: 'tt'}
        self.inverse_label = dict((v,k) for k, v in self.label_map.items())
        
        
    def _load_model(self, model_name, model_download_path, num_classes):
        # define tensor data loading processor
        processor = Preprocess(model_name,
                               cache_dir=model_download_path,
                               batch_size=self.batch_size,
                               n_jobs=self.n_jobs)
        # initialize the model
        model = AutoModelForClassification(model_name, 
                                           model_download_path,
                                           num_classes=num_classes,
                                           finetune_all=self.finetune_all)
        return processor, model
    
        
    def balance(self, train, valid):
        print('Solving Class Imbalance...')
        Balance = BalanceClasses(train, valid, self.random_state)
        train, valid = Balance.experiment2()
        return train, valid
    
        
    def training(self, train, valid, refit=False, fitted_model_path='',
                 warmup=True):
        num_classes = len(set(train.label))
        if self.balance_required:
            train, valid = self.balance(train, valid)
        # load data and encode label
        train = add_info(train)
        valid = add_info(valid)
        train.label = train.label.replace(self.inverse_label)
        valid.label = valid.label.replace(self.inverse_label)
        # define processor and models
        processor, model = self._load_model(self.model_name, 
                                            self.model_download_path, 
                                            num_classes)
        # data to tensor
        train_loader = processor.process_data(train)
        valid_loader = processor.process_data(valid)
        # define trainer
        print('Starting training model...')
        print('using: ' + str(torch.get_num_threads()) + ' cores')
        self.trainer = TrainModel(model, train_loader, valid_loader, self.label_map,
                                  self.batch_size, num_classes, self.max_epochs, 
                                  self.learning_rate, self.random_state, 
                                  refit=refit, fitted_model_path=fitted_model_path,
                                  warmup=warmup)
        
        # training & testing
        self.trainer.train_model()
        self.trainer.save_model(self.model_save_path)


    def testing(self, test_df):
        self.tester = TestModel(self.model_name,
                                self.model_save_path, 
                                self.model_download_path)
        self.tester.test(test_df)


    def save_report(self, refit=False):
        os.makedirs(self.report_save_path,exist_ok=True)
        writer = pd.ExcelWriter(self.report_save_path + 'output.xlsx',
                                engine='xlsxwriter')
        parameters = self._log_parameter()
        parameters.to_excel(writer,sheet_name='parameter_report')
        if refit==False:
            self.tester.test_report.to_excel(writer,sheet_name='test_report')
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
                     }
        df = pd.DataFrame(parameters, index=['Parameters']).T
        df.Parameters = df.Parameters.astype(str)
        return df
         
          
if __name__ == '__main__': 

    # training baselines, input df cols: text, label
    # load data
    train = pd.read_csv('data/train.csv', index_col=0).sample(10000)
    valid = pd.read_csv('data/valid.csv', index_col=0).sample(10000)
    test = pd.read_csv('data/test.csv', index_col=0).sample(10000)
    # some config
    n_jobs = 8 # for controlling cpu
    model_name = 'bert-base-chinese'  
    # training 
    experiment_name = 'bert-base-chinese_temp'
    model = BertTrainer(model_name, experiment_name, n_jobs=n_jobs)
    model.training(train, valid)
    # testing
    model.testing(test) 
    model.save_report()
    
    # #%% the rest is the pipeline for refitting & prediction
    # # loading unused data(example: using more test/valid set) for training
    # train, valid_refit = split_df(test,0.125,42)
    # train = pd.concat([valid, train])
    # # change experiment_name to make a different output path 
    # experiment_name = 'bert-base-chinese_refit_test'
    # # define a model to load
    # best_model = '../nlp_sa_test/output/current_best_balanced/model.pth' 
    # model_refit = BertTrainer(model_name, experiment_name, n_jobs=n_jobs)
    # # reconfig
    # model_refit.balance_required = False
    # model_refit.max_epochs = 2
    # # training
    # model_refit.training(train, valid_refit, refit=True, fitted_model_path=best_model,
    #                      warmup=False)
    # refit_report = model_refit.trainer.valid_report
    # # test on previous model to see if improved
    # pretrained_path = '../nlp_sa_test/pretrained/bert-base-chinese'
    # final_model_path = '../nlp_sa_test/output/current_best_balanced/model.pth'
    # refit_test = TestModel(model_name, final_model_path, pretrained_path, n_jobs=n_jobs)
    # refit_previous_report = refit_test.test(valid_refit)
    # model_refit.save_report(refit=True)
    # refit_previous_report.to_excel('output/bert-base-chinese_refit_test/output_refit_previous_best.xlsx')
    # #%%
    # # prediction execpt 'baby cos' for final deployment using desired model.pth 
    # torch.cuda.empty_cache()
    # pre = pd.read_csv('data/pre.csv',index_col=0)
    # final_model_path = '../nlp_sa_test/output/bert-base-chinese_refit_test/model.pth'
    # final = TestModel(model_name, final_model_path, pretrained_path, n_jobs=n_jobs)
    # pre_cleaned = final.adjust_df(pre)
    # res = final.deployment(pre)
    # final = release(pre_cleaned, res)
    # final.to_csv('sa_202312_release.csv')
     
    
    # ck = pd.read_csv('D:/niq/nlp_sa_test/sa_202312_release.csv').set_index(['shop_id', 'sa_id', 'source_type',])
    # all_zeros_rows = ck[(ck == 0).all(axis=1)]
    
    
    