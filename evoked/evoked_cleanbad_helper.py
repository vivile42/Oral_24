#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 09:41:50 2021

@author: leupinv
"""
import mne
import evoked.evoked_constants as ev_cs
import os
import numpy as np

class EpochCleaner():
    def __init__(self,file,g_num,dict_clean):
        self.epoch=mne.read_epochs(file)
        self.dict_clean=dict_clean
        self.get_type_epo(file)
        
    def get_type_epo(self,file):
        if 'vep' in file:
            self.type_epo='vep'
        else:
            self.type_epo='hep'
    
    def mark_bads(self):
        bad_chann=self.dict_clean[self.type_epo]
        if bad_chann != None:
            self.epoch.info['bads']=bad_chann
        return bad_chann
    
    def interpol_bads(self):
        self.epoch.interpolate_bads()
    
    def save_epoch(self,file):
        self.epoch.save(file, overwrite=True)

        

    
    
            
    
   