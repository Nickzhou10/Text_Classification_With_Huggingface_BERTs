# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 16:24:39 2023

@author: nick
"""

import pandas as pd
import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification as AutoModelSelect


class AutoModelForClassification(nn.Module):
    def __init__(self, model_name, cache_dir, num_classes=7, finetune_all=False):
        super(AutoModelForClassification, self).__init__()
        self.model = AutoModelSelect.from_pretrained(model_name,
                                                     cache_dir=cache_dir, 
                                                     output_hidden_states=True)
        print(self.model)
        self.dropout = nn.Dropout(0.1)
        self.num_classes = num_classes
        # Define a fully connected layer
        self.classifier = torch.nn.Linear(self.model.config.hidden_size, 
                                          self.num_classes)
        # whether to update weights 
        for param in self.model.parameters():
            param.requires_grad = finetune_all
        for param in self.classifier.parameters():
            param.requires_grad = True


    def forward(self, input_ids, attention_mask):
        # run model with the data
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # get hidden layers
        hidden_states = outputs.hidden_states
        # extract the last hidden layer only
        pooler = hidden_states[-1]
        # pooler = torch.mean(pooler, dim=1) 
        # dropout
        pooler = self.dropout(pooler)
        # apply the defined classifier
        output = self.classifier(pooler)
        # tensor shape [batch size, seq length, num_class] to [batch size, num_class]
        output = output.narrow(1, 0, 1)
        output = torch.squeeze(output, dim=1)
        # no need to add an activation function if using nn.CrossEntropyLoss()
        # but will need to add one if in model.eval()
        return output
    
    
if __name__ == '__main__':
    x = torch.randn(3,4,2)
    output = x.narrow(1, 0, 1)
    output = torch.squeeze(output, dim=1)
    print(output.shape)



















