# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 16:24:39 2023

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
from transformers import AutoModelForSequenceClassification
from transformers.utils import logging
# only output error messages
logging.set_verbosity_error()


class AutoModelForClassification(nn.Module):
    """ 
    Initialize the AutoModelForClassification module.

    Args:
        model_name (str): The model name from the Hugging Face model hub.
        cache_dir (str): The directory to save the pre-trained model.
        num_classes (int): The number of classes for the classification task.
        finetune_all (bool): Whether to finetune all layers of the model.
    """
    
    def __init__(self, config):
        super(AutoModelForClassification, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            cache_dir=config.pretrained_cache_path, 
            output_hidden_states=True,
            trust_remote_code=config.trust_remote_code)
        # print(self.model)
        self.dropout = nn.Dropout(config.dropout_ratio)
        # Define a fully connected layer
        self.classifier = torch.nn.Linear(self.model.config.hidden_size, 
                                          config.num_classes)
        # whether to update weights 
        for param in self.model.parameters():
            param.requires_grad = config.enable_finetune_all


    def forward(self, input_ids, attention_mask):
        """
        Perform a forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): The input token IDs of shape (batch_size, sequence_length).
            attention_mask (torch.Tensor): The attention mask of shape (batch_size, sequence_length).
        
        Returns:
            output (logits) (torch.Tensor): The output logits of the 
            model of shape (batch_size, num_classes).
        """
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



















