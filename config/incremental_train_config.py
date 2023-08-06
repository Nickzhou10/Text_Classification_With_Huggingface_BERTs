# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 17:45:56 2023

@author: zhni2001
"""


class Config:
    # data parameters
    random_state = 42
    num_classes = 7
    train_pct = 0.7
    vali_split_ratio = 0.2
    enable_balance = False
    balance_method = 'auto'
    deployment = False
    
    # training device settings
    n_jobs = 8
    device = 'cuda' # 'cpu'
    
    # preprocessing settings
    balance_required = False
    
    # model training settings
    incremental_training = False
    experiment_name = r'test'
    model_name = r'bert-base-chinese'
    batch_size = 64
    learning_rate = 1e-5
    max_epochs = 1
    dropout_ratio = 0.1
    enable_finetune_all = True
    
    # warmup settings
    warmup = True
    scheduler_method = 'linear'
    step_ratio = 0.2
    
    # early stopping settings
    early_stopping = True
    patience = 1

    # data paths
    output_path = r'saved_model/output/' + experiment_name
    model_output_path = output_path  + '/model.pth'
    pretrained_cache_path = r'saved_model/pretrain_model'
    
    # prediction settings
    best_model = 'saved_model/output/bert-base-chinese_refit_test'
    
    
    
    
if __name__ == '__main__':
    
    print(Config.finetune_all)























