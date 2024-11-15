#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 08:36:09 2022

@author: leupinv
"""
import pandas as pd 
import mne



class DropFinder():
    def __init__(self,files_df,files_meta):
        self.df=pd.read_feather(files_df.condition_files[0])
        self.epo=mne.read_epochs(files_meta.condition_files[0])
        self.g_n=files_df.g_num
        
    def filter_df(self):
        self.df=self.df[self.df['difficulty']=='normal']
        self.df=self.df[self.df['accuracy']=='correct']
        