# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 16:44:35 2023

@author: zhni2001
"""
import time
import pandas as pd
import numpy as np
from clean_tools.get_raw import initial_cleaning
from clean_tools.data_chker import checker
from clean_tools.data_cleaner import cleaner


class clean_process:
    
    def __init__(self,data):
        self.data = data
        
        
    def data_ck(self,data,bins):
        ck = checker(data)
        # check number of rows in each class
        ck.check_classes
        # check if there is duplicated rows/same text diff labels
        duplicated = ck.check_duplicates
        # set bins for histgrams
        ck.set_bins = bins
        # check text length by classes
        len_by_bins = ck.check_len_by_bins
        return duplicated,len_by_bins
        

    def clean(self,data): 
        st= time.time()
        clean = cleaner(data)
        # clean text contains 无名
        clean.clean_missing
        # clean rows with numbers only 
        clean.clean_numbers
        # replace characters to meaningful info
        clean.replace_characters
        # solve same text diff labels problems,applied multi processing
        clean.clean_duplicates
        # clean special characters and lengthy texts with length >= x, e.g., remove english >= 25 words
        clean.clean_long_text(25)
        # clean strange rows  
        clean.clean_strange_rows
        # clean empty rows  
        clean.final_clean
        print('time used for cleaning: '+str(time.time()-st))
        return clean.data
    
    
if __name__ == '__main__':
    
    pass
    
    
    
    
    
    
    
    