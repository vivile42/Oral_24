#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 08:35:58 2022

@author: leupinv
"""
import base.files_in_out as files_in_out
import base.base_constants as b_cs
import fixes.find_drop.find_drop_helper as hp
import fixes.find_drop.find_drop_constants as cs
import pandas as pd

for g_n in b_cs.G_N[:1]:
    for cond in b_cs.conditions[0]:
        files_df=files_in_out.GetFiles(filepath=cs.get_filepath(g_n),
                              condition=cond,g_num=g_n,
                               eeg_format=cs.eeg_format_df)
        files_meta=files_in_out.GetFiles(filepath=cs.get_filepath(g_n,epoch=True),
                              condition=cond,g_num=g_n,
                               eeg_format=cs.eeg_format_meta)
        
        drop_finder=hp.DropFinder(files_df,files_meta)
        
       
        
        

        
        