#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 14:33:39 2022

@author: leupinv
"""
    
import os
import platform



platform.system()

# define starting datafolder 

if platform.system()=='Darwin':
    #os.chdir('/Users/leupinv/switchdrive/BBC/WP1/data/EEG/tsk/')
    os.chdir('/Volumes/BBC/BBC/WP1/data/EEG/')
    dir_fold=os.getcwd()
elif platform.system()=='Windows':
    os.chdir('Z:/BBC/WP1/data/EEG/')
    dir_fold=os.getcwd()
   # os.chdir('E:/BBC/WP1/data/EEG/tsk/')
    base_datafolder='E:/'
elif platform.system()=='Linux':
    os.chdir('Z:/BBC/WP1/data/EEG/')
    dir_fold=os.getcwd()
    

eeg_format='rsp_sig.feather'
cond='n'

hiearch_path='tsk/ana/behavioral/hierarch_df_filt.feather'

out_file='tsk/ana/behavioral/behav_df.feather'