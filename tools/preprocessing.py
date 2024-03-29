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

    """
    
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name,
                                                       cache_dir=self.config.pretrained_cache_path,
                                                       trust_remote_code=self.config.trust_remote_code)
        self.drop_last = False
        self.pin_memory = False
        self.shuffle = True
    
    
    def process_pred(self, text):
        pred = self.tokenizer(text, padding=True, max_length=512,
                                truncation=True, return_tensors="pt")
        pred_loader = TensorDataset(pred.input_ids, pred.attention_mask)
        pred_loader = DataLoader(pred_loader)
        return pred_loader
    
    
    def process_data(self, dataset):
        df = dataset.copy()
        tensor_df = self._to_tensor(df)
        loaded_df = self._data_loader(tensor_df)
        return loaded_df
        
    
    def _to_tensor(self, df):
        text = df.text.tolist()
        tokens = self.tokenizer.batch_encode_plus(text,
                                                  padding='longest',
                                                  truncation=True,
                                                  add_special_tokens=True,
                                                  return_attention_mask=True,
                                                  return_token_type_ids=False,
                                                  return_tensors='pt') # output in tensor format
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        labels = torch.tensor(df.label.tolist())
        tensor_df = TensorDataset(input_ids, attention_mask, labels)
        return tensor_df
    
    
    def _data_loader(self, tensor_df):
        dataloader = DataLoader(tensor_df, 
                                batch_size=self.config.batch_size, 
                                shuffle=self.shuffle, 
                                pin_memory=self.pin_memory,
                                num_workers=self.config.n_jobs, 
                                drop_last=self.drop_last)
        return dataloader
    

if __name__ == '__main__':
    # unit test
    from tools.common_tools import label_to_id
    valid = pd.read_csv('../data/pre.csv', index_col=0).rename(columns={'std_name':'text'})
    cache_dir = 'pretrained/bert-base-chinese'
    # valid.label, fitted_label = label_to_id(valid.label)
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
    











