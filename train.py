# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 17:28:24 2023
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
    Training model following a pytorch framework
    
    
    '''
    def __init__(self, model, config):
        self.config = config
        self.device = torch.device(self.config.device)
        torch.manual_seed(self.config.random_state)
        if self.config.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.manual_seed(self.config.random_state)
        self.model = model
        # define optimizer & criterion
        self.optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
        self.criterion = nn.CrossEntropyLoss() 
        # self.class_weights = torch.FloatTensor([...]).to(self.device)
        # self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        # self.criterion = FocalLoss(alpha=self.class_weights) 
        

    def _warmup(self, train_loader):
        self.num_warmup_steps = self.config.max_epochs * \
            len(train_loader) * self.config.step_ratio
        self.num_training_steps = self.config.max_epochs * \
            len(train_loader) 
        self.scheduler = get_scheduler(self.config.scheduler_method,
                                       self.optimizer, 
                                       self.num_warmup_steps, 
                                       self.num_training_steps)
        

    def train_model(self, train_loader, valid_loader):
        # load model / load previous model parameter to model
        if self.config.incremental_training:
            self.model.load_state_dict(torch.load(self.config.fitted_model_path,
                                             map_location=self.device))
            print('incremental training on model: ' + self.config.fitted_model_path)
        self.model = self.model.to(self.device)
        if self.config.warmup:
            self._warmup(train_loader)
        best_valid_loss = float('inf')
        counter = 0
        start = time.time()
        for epoch in range(self.config.max_epochs):
            start_time = time.time()
            current_lr = self.optimizer.param_groups[0]['lr']
            train_loss, self.train_report = self.train(train_loader)
            valid_loss, self.valid_report = self.evaluate(valid_loader)
            time_used = round(time.time() - start_time,1)
            self.epochs_used = epoch
            print(f'Epoch {epoch + 1}/{self.config.max_epochs}, Learning rate: {current_lr}')
            print(f'Training loss: {train_loss:.4f}')
            print(self.train_report)
            print(f'Validation loss: {valid_loss:.4f}')
            print(self.valid_report)
            print(f"Used {time_used} seconds \n")
            # early stopping
            if self.config.early_stopping:
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    counter = 0
                    best_model_weights = self.model.state_dict()
                    self.epochs_used = epoch
                else:  
                    counter += 1
                    if counter >= self.config.patience:
                        print('Early stopping.')
                        break
        # log total time used
        self.total_time_used = round(time.time() - start,1)
        # load weights of the best model if early stopped
        if self.config.early_stopping:
            self.model.load_state_dict(best_model_weights)
        
    
    def _vectorize_label(self,labels):
        # labels to one-hot encoded vectors when necessary
        labels = F.one_hot(labels, num_classes=self.num_classes)
        labels = labels.to(self.device)
        return labels

    
    def _to_device(self, input_ids, attention_mask, labels, device):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        return input_ids, attention_mask, labels
        

    def train(self, train_loader):
        self.model.train()
        total_loss = 0
        predicted = []
        true_labels = []
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
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
            if self.config.warmup:
                self.scheduler.step()
        train_loss = total_loss / len(train_loader)
        res_tb = class_report(true_labels, predicted, 
                              self.config.mapping_dict['mapping'])
        return train_loss, res_tb
    
    
    def evaluate(self, valid_loader):
        self.model.eval()
        total_loss = 0
        predicted = []
        true_labels = []
        with torch.no_grad():
            for batch_idx, (input_ids, attention_mask, labels) in enumerate(valid_loader):
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
                
        valid_loss = total_loss / len(valid_loader)
        res_tb = class_report(true_labels, predicted, 
                              self.config.mapping_dict['mapping'])
        return valid_loss, res_tb
    
    
    def predict(self, model, pred_loader):
        model.eval()
        predicted = []
        with torch.no_grad():
            for batch_idx, (input_ids, attention_mask) in enumerate(pred_loader):
                outputs = model(input_ids, attention_mask=attention_mask)
                outputs = torch.softmax(outputs, dim=1)
                predicted.extend(outputs.cpu().numpy())
        return pd.DataFrame(predicted, columns=range(len(predicted[0])))
    
    
    def save_model(self, output_path):
        torch.save(self.model.state_dict(), output_path)  


if __name__ == '__main__':
    torch.cuda.empty_cache()
    torch.cuda.memory_summary()
    pass


















