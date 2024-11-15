#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 14:33:12 2022

@author: leupinv
"""
import base.base_constants as b_cs
from base.files_in_out import GetFiles,filter_list,getListOfFiles
import fixes.fix_breathing_int.fix_breath_helper as hp
import fixes.fix_breathing_int.fix_breath_constants as cs 



df_list=[]
hier_df=hp.get_hierDF(cs.hiearch_path)
for g_n in b_cs.G_N:
    if g_n=='g23':
        
        filepath_df=f'tsk/preproc/{g_n}/{g_n}_phy_sig'
        rsp_df_files=GetFiles(filepath=filepath_df,
                                  condition=cs.cond,g_num=g_n,
                                  eeg_format=cs.eeg_format)
        filepath_raws=f'raw_nods/{g_n}/'
        
        raw_files=GetFiles(filepath=filepath_raws,
                                  condition=cs.cond,g_num=g_n,
                                  eeg_format='.bdf')
        raw_files.get_info(0)
        rsp_fix=hp.RspFixer(rsp_df_files,raw_files)
        
        # get values relative to inh
        rsp_fix.find_inh_duration()
        rsp_fix.find_rate()
        # filter whole df based on g_num
        hier_df_g=hier_df[hier_df['g_num']==g_n]
        #merge inh values with new df and get final
        def_df=rsp_fix.merge_dfs(hier_df_g,dropna=True)
        
        df_list.append(def_df)

#hp.save_df(df_list)
        

    
    
    