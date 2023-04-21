# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 14:59:30 2023

@author: zhni2001
"""

class DataCleaner:
   def __init__(self, data_source):
       self.data_source = data_source

   def load_data(self):
       # 加载数据
       pass

   def _clean_missing_data(self):
       # 清洗缺失数据
       pass

   def _clean_duplicate_data(self):
       # 清洗重复数据
       pass

   def _clean_invalid_data(self):
       # 清洗无效数据
       pass

   def clean_data(self):
       # 清洗数据
       self._clean_missing_data()
       self._clean_duplicate_data()
       self._clean_invalid_data()

   def save_data(self, output_path):
       # 保存数据
       pass

   @staticmethod
   def log_data_cleaning(func):
       """
       日志记录修饰器
       """
       def wrapper(*args, **kwargs):
           print(f"Cleaning data using {func.__name__} method.")
           result = func(*args, **kwargs)
           print("Data cleaning complete.")
           return result
       return wrapper

   @log_data_cleaning
   def _clean_missing_data(self):
       # 清洗缺失数据
       pass

   @log_data_cleaning
   def _clean_duplicate_data(self):
       # 清洗重复数据
       pass

   @log_data_cleaning
   def _clean_invalid_data(self):
       # 清洗无效数据
       pass