# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 17:45:56 2023

@author: zhni2001
"""


class PredictionConfig:
    model_name = r'bert-base-chinese'
    model_state_dict_path = r'E:/nlp_models/sa_bert_deployment/final_version/saved_model/trained_model/bert-base-chinese_refit_test/model.pth'
    pretrained_cache_path = r'E:/nlp_models/sa_bert_deployment/nlp_sa_test/pretrained/bert-base-chinese'
    device = 'cpu'
    label_map_path = r'E:/nlp_models/sa_bert_deployment/final_version/saved_model/trained_model/bert-base-chinese_refit_test/label_map.pkl'
    
    
if __name__ == '__main__':
    
    print(PredictionConfig.device)























