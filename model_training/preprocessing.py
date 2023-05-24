# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 00:47:31 2023
@author: nick
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from tools.common_tools import label_to_id


class Preprocess:

    def __init__(self, model_name,cache_dir,batch_size=32):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir,trust_remote_code=True)
        self.n_jobs = torch.get_num_threads()-2
        self.batch_size = batch_size
        self.drop_last = False
        self.pin_memory = False
        
    
    def standard_process(self, dataset):
        df = dataset.copy()
        # tokenizing text
        tensor_df = self._tokenization(df)
        # label&text to data loader, controled by batch_size
        loaded_df = self._data_loader(tensor_df)
        return loaded_df


    def _tokenization(self, df):
        text = df.text.tolist()
        labels = torch.tensor(df.label.tolist())
        tokens = self.tokenizer.batch_encode_plus(text,
                                                  padding='longest',
                                                  truncation=True,
                                                  add_special_tokens=True,
                                                  return_attention_mask=True,
                                                  return_token_type_ids=False,
                                                  return_tensors='pt') # output in tensor format
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        tensor_df = TensorDataset(input_ids, attention_mask, labels)
        return tensor_df


    def _data_loader(self,df):
        dataloader = DataLoader(df, batch_size=self.batch_size, 
                                shuffle=True, 
                                pin_memory=self.pin_memory,
                                num_workers=self.n_jobs,
                                drop_last=self.drop_last)
        return dataloader
    
    
if __name__ == '__main__':

    valid = pd.read_csv('data/valid.csv', index_col=0).head(500)
    
    valid.label, fitted_label = label_to_id(valid.label)
    model_name = 'Davlan/distilbert-base-multilingual-cased-ner-hrl'
    processor = Preprocess(model_name)
    valid_dl = processor.standard_process(valid)
    
    for batch_idx, (input_ids, attention_mask, labels) in enumerate(valid_dl):
        batch_input_ids = input_ids
        batch_attention_mask = attention_mask
        batch_labels = labels

    














