# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 17:28:24 2023
https://huggingface.co/Davlan/distilbert-base-multilingual-cased-ner-hrl
@author: nick
"""

import time,os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import get_scheduler
from sklearn.metrics import classification_report
from model import AutoModelForClassification
from tools.common_tools import label_to_id, add_info, class_report
from tools.focal_loss import FocalLoss
import torch.nn.functional as F
import xlsxwriter
 

class TrainModel:
    '''
    give me sometime to edit this 
    
    '''
    def __init__(self, model, train_loader, valid_loader, label_map,
                 batch_size, num_classes, max_epochs, lr, random_state, 
                 refit=False, fitted_model_path='', warmup=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if refit:
            model.load_state_dict(torch.load(fitted_model_path,
                                             map_location=self.device))
            print('refit on model: '+fitted_model_path)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.label_map = label_map
        self.num_classes = num_classes
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping = False  
        self.patience = 1
        self.criterion = nn.CrossEntropyLoss() 
        # self.class_weights = torch.FloatTensor([1/5,1/4,1/1,1/11,
        #                                         1/3,1/1,1/10]).to(self.device)
        # self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        # self.criterion = FocalLoss(alpha=self.class_weights) 
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        self.warmup = warmup
        self.num_warmup_steps = ''
        self.num_training_steps = ''
        self.scheduler_method = ''
        self.random_state = random_state
        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed(self.random_state)


    def _warmup(self, step_ratio=0.2, scheduler_method='linear'):
        if self.warmup:
            self.num_warmup_steps = self.max_epochs * len(self.train_loader) * step_ratio
            self.num_training_steps = self.max_epochs * len(self.train_loader) 
            self.scheduler_method = scheduler_method
            self.scheduler = get_scheduler(self.scheduler_method,
                                            self.optimizer, 
                                            self.num_warmup_steps, 
                                            self.num_training_steps)
        else:
            pass
        

    def train_model(self):
        self._warmup()
        best_valid_loss = float('inf')
        counter = 0
        start = time.time()
        for epoch in range(self.max_epochs):
            start_time = time.time()
            current_lr = self.optimizer.param_groups[0]['lr']
            train_loss, self.train_report = self.train()
            valid_loss, self.valid_report = self.evaluate()
            time_used = round(time.time() - start_time,1)
            self.epochs_used = epoch
            print(f'Epoch {epoch + 1}/{self.max_epochs}, Learning rate: {current_lr}')
            print(f'Training loss: {train_loss:.4f}')
            print(self.train_report)
            print(f'Validation loss: {valid_loss:.4f}')
            print(self.valid_report)
            print(f"Used {time_used} seconds \n")
            # early stopping
            if self.early_stopping:
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    counter = 0
                    best_model_weights = self.model.state_dict()
                    self.epochs_used = epoch
                else:  
                    counter += 1
                    if counter >= self.patience:
                        print('Early stopping.')
                        break
        # log total time used
        self.total_time_used = round(time.time() - start,1)
        # load weights of the best model if early stopped
        if self.early_stopping:
            self.model.load_state_dict(best_model_weights)
        
    
    def _vectorize_label(self,labels):
        # labels to one-hot encoded vectors when necessary
        labels = F.one_hot(labels, num_classes=self.num_classes)
        labels = labels.to(self.device)
        return labels

    
    def _to_device(self, input_ids, attention_mask, labels, device):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        # one-hot if output shape is [batch size, seq length, num_class]
        # labels = self._vectorize_label(labels)
        labels = labels.to(device)
        return input_ids, attention_mask, labels
        

    def train(self):
        self.model.train()
        total_loss = 0
        predicted = []
        true_labels = []
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(self.train_loader):
            input_ids, attention_mask, labels = self._to_device(input_ids,
                                                                attention_mask,
                                                                labels,
                                                                self.device)
            # clean previous gradient calculation
            self.optimizer.zero_grad()
            # get model output
            outputs = self.model(input_ids, attention_mask=attention_mask)
            # calculate loss
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()
            outputs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            predicted.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            loss.backward()
            self.optimizer.step()
            if self.warmup:
                self.scheduler.step()
        train_loss = total_loss / len(self.train_loader)
        res_tb = class_report(true_labels, predicted, self.label_map)
        return train_loss, res_tb
    
    
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        predicted = []
        true_labels = []
        with torch.no_grad():
            for batch_idx, (input_ids, attention_mask, labels) in enumerate(self.valid_loader):
                input_ids, attention_mask, labels = self._to_device(input_ids,
                                                                    attention_mask,
                                                                    labels,
                                                                    self.device)
                # get model output
                outputs = self.model(input_ids, attention_mask=attention_mask)
                # calculate loss
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                outputs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                predicted.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                
        valid_loss = total_loss / len(self.valid_loader)
        res_tb = class_report(true_labels, predicted, self.label_map)
        return valid_loss, res_tb
    
    
    def save_model(self,output_path):
        torch.save(self.model.state_dict(), output_path)  


if __name__ == '__main__':
    torch.cuda.empty_cache()
    torch.cuda.memory_summary()
    pass


















