#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 08:40:15 2022

@author: leupinv
"""


import os
import platform


platform.system()

# define starting datafolder

if platform.system() == 'Darwin':
    #os.chdir('/Users/leupinv/switchdrive/BBC/WP1/data/EEG/tsk/')
    os.chdir('/Volumes/BBC/BBC/WP1/data/EEG/')
    dir_fold = os.getcwd()
elif platform.system() == 'Windows':
    os.chdir('Z:/BBC/WP1/data/EEG/')
    dir_fold = os.getcwd()
   # os.chdir('E:/BBC/WP1/data/EEG/tsk/')
    base_datafolder = 'E:/'
elif platform.system() == 'Linux':
    os.chdir('Z:/BBC/WP1/data/EEG/')
    dir_fold = os.getcwd()


eeg_format = 'epo.fif'
cond = 'n_tsk_cfa_vep'




def get_filepath(g_n,epoch=False):
    
    filepath = f'tsk/preproc/{g_n}/{g_n}_mrk_DF'
    if epoch:
        filepath = f'tsk/preproc/{g_n}/{g_n}_epochs'
    
    return filepath

eeg_format_df='stim_mrk.feather'

eeg_format_meta='cfa_vep_clean_epo.fif'