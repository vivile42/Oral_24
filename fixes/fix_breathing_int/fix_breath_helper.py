#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 14:33:27 2022

@author: leupinv
"""
import pandas as pd
import feather
from markers.markers_MNE_helper import MarkersMNE
import fixes.fix_breathing_int.fix_breath_constants as cs 


def get_hierDF(path):
    df=pd.read_feather(path)
    df=df[df["signal_type"]=='vep']
    return df

def save_df(df_list):
    behav_df=pd.concat(df_list)
    
    feather.write_dataframe(behav_df, cs.out_file)
    

class RspFixer():
    def __init__(self,files_rsp,files_raw):
        self.g_n=files_rsp.g_num
        # Load rsp df
        self.df=pd.read_feather(files_rsp.condition_files[0])
        # Filter out peaks
        self.df=self.get_peaks_df(self.df)
        
        #get srate
        
        self.get_sr(files_raw)
        self.save_sr()
    
    def get_peaks_df(self,df):
        df=df.loc[(df['RSP_Peaks']==1) | 
                            (df['RSP_Troughs']==1)]
        df.reset_index(inplace=True)
        df.rename(dict(index='TF'),axis=1,inplace=True)
        return df
        
    def find_rate(self):
        Rsp_rate_df=self.df[self.df['RSP_Troughs']==1]
        Rsp_rate_df.drop(['RSP_Peaks','RSP_Troughs'],axis=1,inplace=True)
        
        Rsp_rate_df['rsp_int_inh']=(Rsp_rate_df['TF'].diff().shift(-1))/self.sr
        self.rsp_rate=Rsp_rate_df[['TF','rsp_int_inh','inh_dur']]
    
    def find_inh_duration(self):
        self.df['inh_dur']=(self.df['TF'].diff().shift(-1))/self.sr
        
    
    def get_sr(self,files_raw):
        mne_data=MarkersMNE(files_raw)
        self.sr=mne_data.srate
        
    def save_sr(self):
        filename=f'raw_nods/{self.g_n}/{self.g_n}_tsk_n_info.txt'
        with open(filename,'w') as file:
            file.write(f'srate= {self.sr}')
        
    def merge_dfs(self,hierDF,dropna=True):
        '''
        

        Parameters
        ----------
        hierDF : TYPE PandasDF
            DESCRIPTION. df containing stimulus markers
        dropna : TYPE, optional
            DESCRIPTION. Whether  you want to filter out inh exh markers,
            The default is True.

        Returns
        -------
        def_df : TYPE Pandas DF
            DESCRIPTION. merged database with new rsp parametres

        '''
        def_df=hierDF.merge(self.rsp_rate,how='outer')
        def_df.sort_values('TF',inplace=True)
        def_df[['rsp_int_inh','inh_dur']]=def_df[['rsp_int_inh','inh_dur']].fillna(method='pad')
        if dropna:
            def_df.dropna(inplace=True)
        return def_df
        
        
