# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 00:47:31 2023
@author: nick
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer


class Preprocess:
    """ 
    It is designed to processes df to tensor df automatically. 
    All you need is just to input the name of the pretrained model 
    you wish to use and the download path to save the downloaded tokenizer 
    for reuse.

    Shape:
        - dataset: df with df.label and df.text
        - dataset for deployment process: df with df.text and set deployment=True

    Args:
        model_name (str, required): model name copied from huggingface 
        cache_dir (str, required): place to save the pre-trained model
        batch_size (int, optional): batch_size for DataLoader.
            Defaults to 32.
        n_jobs (int, optional): cpu cores to use for DataLoader.
            Defaults to 8.
    """
    
    def __init__(self, model_name, cache_dir, batch_size=32, n_jobs=8):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       cache_dir=cache_dir,
                                                       trust_remote_code=True)
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.drop_last = False
        self.pin_memory = False
        
    
    def process_data(self, dataset, deployment=False):
        df = dataset.copy()
        text = df.text.tolist()
        labels = torch.tensor(df.label.tolist())
        info = torch.tensor(df.index.tolist())
        tokens = self.tokenizer.batch_encode_plus(text,
                                                  padding='longest',
                                                  truncation=True,
                                                  add_special_tokens=True,
                                                  return_attention_mask=True,
                                                  return_token_type_ids=False,
                                                  return_tensors='pt') # output in tensor format
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        
        if deployment:
            tensor_df = TensorDataset(input_ids, attention_mask, info)
        else:
            tensor_df = TensorDataset(input_ids, attention_mask, labels)
        
        return self._data_loader(tensor_df, shuffle=not deployment)
    
    
    def _data_loader(self, tensor_df, shuffle=True):
        dataloader = DataLoader(tensor_df, batch_size=self.batch_size, 
                                shuffle=shuffle, 
                                pin_memory=self.pin_memory,
                                num_workers=self.n_jobs,
                                drop_last=self.drop_last)
        return dataloader
    
    
if __name__ == '__main__':
    # unit test
    from tools.common_tools import label_to_id
    valid = pd.read_csv('../data/valid.csv', index_col=0).sample(500)
    cache_dir = '../nlp_sa_test/pretrained/bert-base-chinese'
    valid.label, fitted_label = label_to_id(valid.label)
    model_name = 'bert-base-chinese'  
    processor = Preprocess(model_name, cache_dir)
    valid_dl = processor.process_data(valid)
    
    for batch_idx, (input_ids, attention_mask, labels) in enumerate(valid_dl):
        batch_input_ids = input_ids
        batch_attention_mask = attention_mask
        batch_labels = labels
        
    print('batch_input_ids shape: \n',batch_input_ids.shape)
    print('batch_attention_mask shape: \n',batch_attention_mask.shape)
    print('batch_labels shape: \n',batch_labels.shape)
    














