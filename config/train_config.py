# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 17:45:56 2023

@author: zhni2001
"""


class TrainConfig:
    # data parameters
    random_state = 42
    num_classes = 7
    vali_split_ratio = 0.2
    
    
    # training device settings
    n_jobs = 8
    device = 'cpu' # 'cuda'
    
    # preprocessing settings
    balance_required = False
    balance_method = 'auto'
    # model training settings
    incremental_training = False
    experiment_name = r'test_0915'
    model_name = r'bert-base-chinese'
    trust_remote_code = False
    batch_size = 4
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
    output_path = r'saved_model/trained_model/' + experiment_name
    model_output_path = output_path  + '/model.pth'
    pretrained_cache_path = r'saved_model/pretrain_model/' + model_name
     
    
if __name__ == '__main__':
    
    print(TrainConfig.enable_finetune_all)























