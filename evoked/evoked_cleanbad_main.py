#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 09:41:32 2021

@author: leupinv
"""
from base.files_in_out import GetFiles,filter_list

import evoked.evoked_constants as cs
import base.base_constants as b_cs

from evoked.evoked_cleanbad_helper import EpochCleaner
import gc
import mne

# In[Get list of epochs]


for g_n in cs.clean_dict:
    g_n,dict_clean=g_n
    files = GetFiles(filepath=cs.datafolder,
                     condition=cs.condition,g_num=g_n,
                     eeg_format='clean_epo.fif')
    for file in files.condition_files:
        if 'xns'in file:
            continue
        epo_clean=EpochCleaner(file,g_n,dict_clean)
        bads=epo_clean.mark_bads()
        if bads !=None:
            epo_clean.interpol_bads()
            epo_clean.save_epoch(file)
        

        #         if cfa in file:
        #             epochs_group.add_epoch(file,g_num=g_n,idx=idx)
            
        # epochs_group.get_all_list()
        # epochs_group.get_averages(cfa)
        # del epochs_group
        # gc.collect()
