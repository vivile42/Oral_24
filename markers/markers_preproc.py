
#!/usr/bin/env python
# coding: utf-8

# # Partial preprocessing
# 
# the first step of the preprocessing should lead to the downsamplig + extraction of triggers
# 
# Trigger exctrations+ ecg preproc is best done in python while for the downsampling we'll use EEGlab (given better export is available for bdf file
# 
# ## Sections 
# ### [Import](#import)
# Loop into all subdirectories and get file ending with .bdf and including tsk    
# 
# ### [MNE loading](#MNE)
# Load into MNE, assign channel types to then select them
# 
# ### [Stim awareness mrk](#stim_awa)
# select stim channel, format mrk and put them into pandas table to derive awareness condition
# 
# ### [Stim RT](#RT)
# 
# 

# In[2]:
import os
#base=os.getcwd()
os.chdir('c:/Users/Engi/all/BBC/WP1/data/Code/python/tsk_processing')
#os.chdir('/Users/leupinv/switchdrive/BBC/WP1/data/Code/python/tsk_processing')
import mne
import pandas as pd 
import re #regex library to format fast to read into pd 

import neurokit2 as nk 
import matplotlib.pyplot as plt
import feather 
import numpy as np
import platform
from files_in_out import getListOfFiles,GetFiles
from markers_DF import get_DF_centerd,get_help_DF

# In[3]:
    
#import matplotlib
#matplotlib.use('Agg')


platform.system()


# In[4]:


#current main directory 


if platform.system()=='Darwin':
    os.chdir('/Users/leupinv/switchdrive/BBC/WP1/data/EEG/tsk/')
    datafolder='/Volumes/Elements/'
else:
    os.chdir('c:/Users/Engi/all/BBC/WP1/data/EEG/tsk')
    datafolder='E:/'


# In[5]
g_num='g01'
eeg_format='bdf'
eeg_exp='tsk'
condition=['n','o']

#files = getListOfFiles(datafolder,g_num)

files = GetFiles(datafolder,g_num)

for cond in condition:
    
    files.select_condition(condition=cond)
    

    taskfiles=files.condition_files
    
    ecg_sig_list=[]
    rsp_sig_list=[]
    # In[7]:
    
    for i in range(len(taskfiles)):
        files.get_info(index=i)
        
        dir_tsk=files.current_file_dir
        
        
        #tsk_len=len(datafolder)+4
        
        
# In[8]:
        
# # MNE loading
# Channel selection and basic loading into MNE
        
        
        
        raw=mne.io.read_raw_bdf(dir_tsk, preload=True) 
        raw.set_channel_types({'Erg1':'resp','EXG1':'ecg'})
        montage=mne.channels.make_standard_montage('biosemi128')
        raw.set_montage(montage)
        raw.set_eeg_reference('average',projection=False) 
        
        
        # In[10]:
        
        
        #get sampling rate 
        srate=raw.info['sfreq']
        
        

# # Stimulus awareness markers
        
        # In[12]:
        
        
        events=mne.find_events(raw,consecutive=True,shortest_event=1)
        
        
        # In[13]:
        
        
        events_coded=[[x[0],x[1],256-(2**16-x[2])] for x in events ]
        
        
        # In[14]:
        
        
        mrk=pd.DataFrame(events_coded)
        mrk.columns=['TF','nul','trigger']
        mrk=mrk.drop(columns=["nul"])
        mrk.set_index('TF',inplace=True)
        
        
        
        # In[15]:
        
        
        df_2=mrk.loc[(mrk['trigger']==1)|(mrk['trigger']==2)|(mrk['trigger']==10)|(mrk['trigger']==20)]
        df_3=mrk.loc[(mrk['trigger']==3)|(mrk['trigger']==4)]
        df_5=mrk.loc[(mrk['trigger']==5)|(mrk['trigger']==6)]
        mrk_mrg=pd.concat([df_2,df_3,df_5],axis=1)
        
        
        
        # In[16]:
        
        
        mrk_mrg.reset_index(inplace=True)
        mrk_mrg.columns=['TF','trigger_stim','trigger_corr','trigger_aware']
        
        
        
        # In[17]:
        
        
        #get df to compute state as a function of the response 
        method=['pad','backfill']
        
        df_ans=get_DF_centerd(mrk_mrg.copy(),cent_column='trigger_corr',srate=srate,method=method)
        
        df_ans['difficulty']=['easy' if x==20 or x==10 else 'normal'for x in df_ans['trigger_stim']]
        
        condition=['RCA','RCU','RCE','RUE','LCA','LCU','LCE','LUE','RCA','RCU','RCE','RUE','LCA','LCU','LCE','LUE']
        columns_existing=['trigger_stim','trigger_corr','trigger_aware']
        
        df_h=get_help_DF(df=df_ans,columns_existing=columns_existing,new_column=condition,new_column_name='condition')
        df_ans=df_ans.merge(df_h,on=columns_existing,how='left')
        
        
        # In[18]:
        df_ans['accuracy']=['corr' if x==3 else 'error' for x in df_ans['trigger_corr']] 
        df_ans['awareness']=['aware' if x==5 else 'unaware' for x in df_ans['trigger_aware']] 
            
        
        # In[18]:
        #get df to compute state as a function of the marker
        
        
        df_mrk=get_DF_centerd(mrk_mrg.copy(),cent_column='trigger_stim',srate=srate)
        df_mrk=df_mrk.dropna(how='any',axis=0)
        
        
        # In[21]:
        
        
        df_mrk['difficulty']=['easy' if x==20 or x==10 else 'normal'for x in df_mrk['trigger_stim']]
        df_mrk
        
        
        # # Stim awareness 
        # ## Set condition using helping dataframe
        
        # In[22]:
        
        
        condition=['RCA','RCU','RCE','RUE','LCA','LCU','LCE','LUE','RCA','RCU','RCE','RUE','LCA','LCU','LCE','LUE']
        columns_existing=['trigger_stim','trigger_corr','trigger_aware']
        
        df_h=get_help_DF(df=df_mrk,columns_existing=columns_existing,new_column=condition,new_column_name='condition')
        
        # In[23]:
        
        
        df_mrk=df_mrk.merge(df_h,on=columns_existing,how='left')
        
        
        # ## RSP preprocessing
        
        # In[24]:
        
        
        rsp=raw.copy().pick_types(resp=True)
        
        
        
        resp=rsp.get_data().flatten()*-1
        
        
        rsp_signals,rsp_info=nk.rsp_process(resp,sampling_rate=srate)
        fig_rsp=nk.rsp_plot(rsp_signals,sampling_rate=srate)
        
        rsp_sig_list.append(rsp.get_data())
        
        # In[28]:
        type_sig='png'
        file_end='rsp_fig.png'
        
        output_filename=files.out_filename(type_sig=type_sig,file_end=file_end)    
        
        
        fig_rsp.savefig(output_filename,dpi=2000)
        
        
        
        # In[30]:
        
        
        type_sig='phy_sig'
        file_end='rsp_sig.feather'
        out_filename=files.out_filename(type_sig=type_sig,file_end=file_end)
        feather.write_dataframe(rsp_signals, out_filename)
        
        
        # In[ ]:
        
        
        
        
        
        # In[31]:
        
        
        rsp_signals_pk=rsp_signals[['RSP_Peaks','RSP_Troughs']]
        rsp_signals_pk=rsp_signals_pk.loc[(rsp_signals_pk['RSP_Peaks']==1)|(rsp_signals_pk['RSP_Troughs']==1)]
        rsp_signals_pk
        
        
        # In[32]:
        
        
        rsp_signals_pk['rsp_phase']=['exh' if x ==1 else 'inh' for x in rsp_signals_pk['RSP_Peaks']]
        
        
        rsp_df_mrk=rsp_signals_pk['rsp_phase'].reset_index()
        rsp_df_mrk.columns=['TF','rsp_phase']
        rsp_df_mrk
        
        
        # ## ECG preprocessing
        
        # In[33]:
        
        
        ecg=raw.copy().pick_types(ecg=True).get_data()
        ecg_sig=ecg.flatten()
        
        ecg_sig_list.append(ecg)
        # 
        # 
        
        # In[34]:
        
        
        
        sampling_rate=srate
        ecg_signal = nk.signal_sanitize(ecg_sig)
        
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method='neurokit')
        # R-peaks
        instant_peaks, rpeaks, = nk.ecg_peaks(
            ecg_cleaned=ecg_cleaned, sampling_rate=sampling_rate, method='neurokit', correct_artifacts=False
        )
        rate = nk.signal_rate(rpeaks, sampling_rate=sampling_rate, desired_length=len(ecg_cleaned))
        
        quality = nk.ecg_quality(ecg_cleaned, rpeaks=None, sampling_rate=sampling_rate)
        
        signals = pd.DataFrame({"ECG_Raw": ecg_signal, "ECG_Clean": ecg_cleaned, "ECG_Rate": rate, "ECG_Quality": quality})
        
            # Additional info of the ecg signal
        delineate_signal, delineate_info = nk.ecg_delineate(
            ecg_cleaned=ecg_cleaned, rpeaks=rpeaks, sampling_rate=sampling_rate
        )
        
        cardiac_phase = nk.ecg_phase(ecg_cleaned=ecg_cleaned, rpeaks=rpeaks, delineate_info=delineate_info)
        
        ecg_signals = pd.concat([signals, instant_peaks, delineate_signal, cardiac_phase], axis=1)
        
        info = rpeaks
        ecg_fig = nk.ecg_plot(ecg_signals, sampling_rate=srate)
        
        
        # In[35]:
        
        type_sig='png'
        file_end='ecg_fig.png'
        output_filename=files.out_filename(type_sig=type_sig,file_end=file_end)
        
        ecg_fig.savefig(output_filename,dpi=2000)
        
        
        # In[36]:
        
        
        r_peaks=ecg_signals[ecg_signals['ECG_R_Peaks']==1]
        r_peaks.reset_index(inplace=True)
        r_peaks=r_peaks['index']
        
        
        # In[37]:
        
        
        t_off=ecg_signals[ecg_signals['ECG_T_Offsets']==1]
        t_off.reset_index(inplace=True)
        t_off=t_off['index']
        
        
        # In[38]:
        
        
        
        #nan_values=t_off.isnull()
        #nan_values[nan_values==True]
        
        
        # In[39]:
        
        
        #plot = nk.events_plot(t_off[:100],ecg_signals['ECG_Clean'][:100000] )
        
        
        # In[40]:
        
        
        
        # In[41]:
        file_end='ecg_sig.feather'
        type_sig='phy_sig'
        output_filename=files.out_filename(type_sig=type_sig,file_end=file_end)
        
        feather.write_dataframe(ecg_signals, output_filename)
        
        
        # In[ ]:
        
        
        
        
        
        # In[42]:
        
        
        ecg_signals_pk=ecg_signals[['ECG_R_Peaks','ECG_T_Offsets']]
        ecg_signals_pk=ecg_signals_pk.loc[(ecg_signals_pk['ECG_R_Peaks']==1)|(ecg_signals_pk['ECG_T_Offsets']==1)]
        
        ecg_signals_pk['cardiac_phase']=['R' if x ==1 else 'T' for x in ecg_signals_pk['ECG_R_Peaks']]
        
        
        cardiac_mrk=ecg_signals_pk['cardiac_phase'].reset_index()
        cardiac_mrk.columns=['TF','cardiac_phase']
        
        
        
        # # MRK mrg to get mrk as a function of breathing and cardiac phase 
        
        # In[43]:
        
        
        merged_df=pd.concat([df_mrk,cardiac_mrk,rsp_df_mrk])
        
        
        # ## generate merged long database with all TFs
        # From this database you can derive both stimulus databases and Heps databases
        # 
        
        # In[44]:
        
        
        merged_df.sort_values(by=['TF'],inplace=True)
        merged_df.set_index('TF',inplace=True)
        merged_df['trigger_stim']=merged_df['trigger_stim'].fillna(0)
        merged_df['trigger_corr']=merged_df['trigger_corr'].fillna(0)
        merged_df['trigger_aware']=merged_df['trigger_aware'].fillna(0)
        merged_df['RT']=merged_df['RT'].fillna(method='backfill')
        merged_df['difficulty']=merged_df['difficulty'].fillna(method='backfill')
        merged_df['condition']=merged_df['condition'].fillna(method='backfill')
        merged_df['cardiac_peak']=merged_df['cardiac_phase'].fillna('no_peak')
        merged_df['cardiac_phase']=merged_df['cardiac_phase'].fillna(method='pad')
        merged_df['cardiac_phase']=merged_df['cardiac_phase'].map({'R':'sys','T':'dia'})
        merged_df['cardiac_phase'].fillna(value='no_peak',inplace=True)
        merged_df['rsp_phase']=merged_df['rsp_phase'].fillna(method='pad')
        if rsp_df_mrk['rsp_phase'][0]=='inh':
            merged_df['rsp_phase'].fillna(value='exh',inplace=True)
        else:
             merged_df['rsp_phase'].fillna(value='inh',inplace=True)
    
        
        
        # In[45]:
        
        
        merged_df.dropna(inplace=True)
        merged_df.reset_index(inplace=True)
        
        
        # In[46]:
        
        
        type_sig='stim'
        ile_end='raw.feather'
        output_filename=files.out_filename(type_sig=type_sig,file_end=file_end)
        
        feather.write_dataframe(merged_df, output_filename)
        
        
        # In[ ]:
        
        
        
        
        
        # ## Stim database 
        # Dervie stimulus datavase out of merged
        
        # In[47]:
        
        
        stim_df=merged_df[merged_df['trigger_corr']!=0]
        stim_df
        
        
        
        
        # In[48]:
        
        
        awa=['CA','CE','CU','UE','CA','CE','CU','UE']
        df_h=get_help_DF(stim_df,['condition'],awa,'aware_mrk')
        
        
        
        # In[50]:
        
        
        stim_df=stim_df.merge(df_h,on=['condition'],how='left')
        
        
        
        
        # In[51]:
        
        on_col=['aware_mrk','cardiac_phase']
        cardiac_mrk_h=['CAD','CAS','CED','CES','CUD','CUS','UED','UES']
        
        df_h=get_help_DF(stim_df,on_col,cardiac_mrk_h,'cardiac_mrk')
        
        
        
        #%%
        
        
        # In[52]:
        
        
        stim_df=stim_df.merge(df_h,on=on_col,how='left')
        
        stim_df
        
        
        # In[53]:
        
        on_col=['aware_mrk','rsp_phase']
        
        rsp_mrk=['UEI','UEE','CEI','CEE','CAI','CAE','CUI','CUE']
        
        rsp_mrk.sort()
        
        df_h=get_help_DF(stim_df,on_col,rsp_mrk,'rsp_mrk')
        
        
        
        # In[54]:
        
        
        stim_df=stim_df.merge(df_h,on=on_col,how='left')
        
        
        
        # In[55]:
        
        on_col=['rsp_mrk','cardiac_phase']
        
        rspXcar_mrk=['CADE','CASE','CADI','CASI','CEDE','CESE','CEDI','CESI','CUDE','CUSE','CUDI','CUSI','UEDE','UESE','UEDI','UESI']
        
        
        df_h=get_help_DF(stim_df,on_col,rspXcar_mrk,'rspXcar_mrk')
        
        #%%
        stim_df=stim_df.merge(df_h,on=on_col,how='left')
        
        
        
        # In[56]:
        
        
        type_sig='stim'
        file_end='stim.feather'
        
        output_filename=files.out_filename(type_sig=type_sig,file_end=file_end)
        
        feather.write_dataframe(stim_df, output_filename)
        
        
        
        # In[57]:
        
        
        mrk_df= stim_df[stim_df['difficulty']=='normal']
        
        
        
        # In[58]:
        
        
        mrk_df=mrk_df.loc[(mrk_df['aware_mrk']=='CA')|(mrk_df['aware_mrk']=='CU')]
        
        drop_col=['trigger_stim', 'trigger_corr', 'trigger_aware','difficulty','cardiac_peak']
        mrk_df.drop(drop_col,axis=1,inplace=True)
        
        # In[59]: Answer DF 
        
        
        
            
            
        merged_df_ans=pd.concat([df_ans,cardiac_mrk,rsp_df_mrk])
        
        
        # ## generate merged long database with all TFs
        # From this database you can derive both stimulus databases and Heps databases
        # 
        
        # In[44]:
        
        
        merged_df_ans.sort_values(by=['TF'],inplace=True)
        merged_df_ans.set_index('TF',inplace=True)
        merged_df_ans['trigger_stim']=merged_df_ans['trigger_stim'].fillna(0)
        merged_df_ans['trigger_corr']=merged_df_ans['trigger_corr'].fillna(0)
        merged_df_ans['trigger_aware']=merged_df_ans['trigger_aware'].fillna(0)
        merged_df_ans['RT']=merged_df_ans['RT'].fillna(method='backfill')
        merged_df_ans['difficulty']=merged_df_ans['difficulty'].fillna(method='backfill')
        merged_df_ans['condition']=merged_df_ans['condition'].fillna(method='backfill')
        merged_df_ans['cardiac_peak']=merged_df_ans['cardiac_phase'].fillna('no_peak')
        merged_df_ans['cardiac_phase']=merged_df_ans['cardiac_phase'].fillna(method='pad')
        merged_df_ans['cardiac_phase']=merged_df_ans['cardiac_phase'].map({'R':'sys','T':'dia'})
        merged_df_ans['cardiac_phase'].fillna(value='no_peak',inplace=True)
        merged_df_ans['rsp_phase']=merged_df_ans['rsp_phase'].fillna(method='pad')
        if rsp_df_mrk['rsp_phase'][0]=='inh':
            merged_df_ans['rsp_phase'].fillna(value='exh',inplace=True)
        else:
             merged_df_ans['rsp_phase'].fillna(value='inh',inplace=True)
        
        
        # In[45]:
        
        
        merged_df_ans.dropna(inplace=True)
        merged_df_ans.reset_index(inplace=True)
        
        
        # In[ ]:
        
        
        
        
        
        # ## Stim database 
        # Dervie stimulus datavase out of merged
        
        # In[47]:
        
        
        ans_df=merged_df_ans[merged_df_ans['trigger_corr']!=0]
        ans_df
        
        
        
        
        # In[48]:
        
        
        awa=['NCA','NCE','NCU','NUE','NCA','NCE','NCU','NUE']
        df_h=get_help_DF(ans_df,['condition'],awa,'aware_mrk')
        
        
        
        # In[50]:
        
        
        ans_df=ans_df.merge(df_h,on=['condition'],how='left')
        
        
        
        
        # In[51]:
        
        on_col=['aware_mrk','cardiac_phase']
        cardiac_mrk_h=['NCAD','NCAS','NCED','NCES','NCUD','NCUS','NUED','NUES']
        
        df_h=get_help_DF(ans_df,on_col,cardiac_mrk_h,'cardiac_mrk')
        
        
        
        #%%
        
        
        # In[52]:
        
        
        ans_df=ans_df.merge(df_h,on=on_col,how='left')
        
        
        
        # In[53]:
        
        on_col=['aware_mrk','rsp_phase']
        
        rsp_mrk=['NUEI','NUEE','NCEI','NCEE','NCAI','NCAE','NCUI','NCUE']
        
        rsp_mrk.sort()
        
        df_h=get_help_DF(ans_df,on_col,rsp_mrk,'rsp_mrk')
        
        
        
        # In[54]:
        
        
        ans_df=ans_df.merge(df_h,on=on_col,how='left')
        
        
        
        # In[55]:
        
        # on_col=['rsp_mrk','cardiac_phase']
        
        # rspXcar_mrk=['NCADE','NCASE','NCADI','NCASI','NCEDE','NCESE','NCEDI','NCESI','NCUDE','NCUSE','NCUDI','NCUSI','NUEDE','NUESE','NUEDI','NUESI']
        
        
        # df_h=get_help_DF(ans_df,on_col,rspXcar_mrk,'rspXcar_mrk')
        
        # #%%
        # ans_df=ans_df.merge(df_h,on=on_col,how='left')
        
        
        
        # In[56]:
        
        
        type_sig='ans'
        file_end='ans.feather'
        
        output_filename=files.out_filename(type_sig=type_sig,file_end=file_end)
        
        feather.write_dataframe(ans_df, output_filename)
        
        
        
        # In[57]:
        
        
        ans_df= ans_df[ans_df['difficulty']=='normal']
        
        
        
        # In[58]:
        
        
        drop_col=['trigger_stim', 'trigger_corr', 'trigger_aware','difficulty','cardiac_peak']
        ans_df.drop(drop_col,axis=1,inplace=True)
        
        # In[59]:
        
        
        # import seaborn as sns
        # CU_DF=mrk_df[mrk_df['aware_mrk']=='CU']
        # 
        # removed_out=CU_DF['RT'].between(CU_DF['RT'].quantile(0),CU_DF['RT'].quantile(.95))
        # id_rem=CU_DF[~ removed_out].index
        # df_out=CU_DF.drop(id_rem)
        # 
        # sns.stripplot(x='cardiac_phase',y='RT',data=CU_DF,hue='rsp_phase')
        
        # CA_DF=mrk_df[mrk_df['aware_mrk']=='CA']
        # 
        # removed_out=CA_DF['RT'].between(CA_DF['RT'].quantile(0),CA_DF['RT'].quantile(.95))
        # id_rem=CA_DF[~ removed_out].index
        # df_out=CA_DF.drop(id_rem)
        # 
        # sns.stripplot(x='cardiac_phase',y='RT',data=CA_DF,hue='rsp_phase')
        
        # In[60]:
        
        # # HEP df
        # Create df for HEP as f(x) of awareness + HEP as f(X) of stimulus condition 
        # 
        # Create an allR condition in which you can check R peaks-evoked rsponse
        # 
        # Create a t-wave preceding/R-peak preceding stimulus 
        
        # In[61]:
        
        
        ## need to create DF with t-wave peak instead of offset
        ecg_signals_Tpk=ecg_signals[['ECG_R_Peaks','ECG_T_Peaks']]
        ecg_signals_Tpk=ecg_signals_Tpk.loc[(ecg_signals_Tpk['ECG_R_Peaks']==1)|(ecg_signals_Tpk['ECG_T_Peaks']==1)]
        
        ecg_signals_Tpk['cardiac_phase']=['R' if x ==1 else 'T' for x in ecg_signals_Tpk['ECG_R_Peaks']]
        
        cardiac_mrk_Tpk=ecg_signals_Tpk['cardiac_phase'].reset_index()
        cardiac_mrk_Tpk.columns=['TF','cardiac_phase']
        cardiac_mrk_Tpk
        
        
        # In[ ]:
        
        
        
        
        
        # Generate merged dataframe with T-peaks 
        
        # In[62]:
        
        
        merged_df_Tpk=pd.concat([df_mrk,cardiac_mrk_Tpk,rsp_df_mrk])
        merged_df_Tpk.sort_values(by=['TF'])
        merged_df_Tpk.sort_values(by=['TF']).head(100)
        merged_df_Tpk.sort_values(by=['TF'],inplace=True)
        merged_df_Tpk.set_index('TF',inplace=True)
        merged_df_Tpk['trigger_stim']=merged_df_Tpk['trigger_stim'].fillna('no_stim')
        merged_df_Tpk['trigger_corr']=merged_df_Tpk['trigger_corr'].fillna('no_stim')
        merged_df_Tpk['trigger_aware']=merged_df_Tpk['trigger_aware'].fillna('no_stim')
        merged_df_Tpk['RT']=merged_df_Tpk['RT'].fillna(method='backfill')
        merged_df_Tpk['difficulty']=merged_df_Tpk['difficulty'].fillna(method='backfill')
        merged_df_Tpk['condition']=merged_df_Tpk['condition'].fillna(method='backfill')
        merged_df_Tpk['cardiac_peak']=merged_df_Tpk['cardiac_phase'].fillna('no_peak')
        merged_df_Tpk['rsp_phase']=merged_df_Tpk['rsp_phase'].fillna(method='pad')
        if rsp_df_mrk['rsp_phase'][0]=='inh':
            merged_df_Tpk['rsp_phase'].fillna(value='exh',inplace=True)
        else:
             merged_df_Tpk['rsp_phase'].fillna(value='inh',inplace=True)
        
        merged_df_Tpk.head(5)
        
        
        # In[63]:
        
        
        
        hep_df=merged_df_Tpk.loc[(merged_df_Tpk['cardiac_peak']!='no_peak')|(merged_df_Tpk['trigger_corr']!='no_stim')]
        hep_df=hep_df[(hep_df['trigger_corr']!='no_stim').shift(-1).fillna(False)]
        hep_df.reset_index(inplace=True)
        hep_df
        
        
        # sns.stripplot(y='RT',data=hep_df,x='condition',hue='rsp_phase')
        
        # In[64]:
        
        awa=['CA','CE','CU','UE','CA','CE','CU','UE']
        df_h=get_help_DF(hep_df,['condition'],awa,'aware_mrk')
        
        
        
        # In[50]:
        
        
        hep_df=hep_df.merge(df_h,on=['condition'],how='left')
        
        
        # In[66]:
        on_col=['cardiac_peak','aware_mrk']
        
        awa_card=['TUE','RUE','TCE','RCE','TCA','RCA','TCU','RCU']
        
        awa_card.sort()
        
        df_h=get_help_DF(hep_df,on_col,awa_card,'awareXcard')
        
        
        
        # In[67]:
        
        
        hep_df=hep_df.merge(df_h,on=on_col,how='left')
        
        hep_df
        
        
        # In[68]:
        
        on_col=['rsp_phase','awareXcard']
        
        awa_card_rsp=['RECA','RICA','RECE','RICE','RECU','RICU','REUE','RIUE','TECA','TICA','TECE','TICE','TECU','TICU','TEUE','TIUE']
        
        
        df_h=get_help_DF(hep_df,on_col,awa_card_rsp,'awareXcardXrsp')
        
        
        # In[69]:
        
        
        hep_df=hep_df.merge(df_h,on=on_col,how='left')
        
        hep_df
        
        
        # ### Write output 
        # write datafrme to preproc output for each subject 
        
        # In[70]:
        
        
        
        
        type_sig='hep'
        
        file_end='hep.feather'
        
        output_filename=files.out_filename(type_sig=type_sig,file_end=file_end)
        
        feather.write_dataframe(hep_df, output_filename)
        
        
        
        # In[ ]:
        
        
        
        
        
        # In[71]:
        
        
        #save DF before this point but exclud easy stim for markers 
        mrk_hep_df=hep_df.loc[(hep_df['aware_mrk']=='CA')|(hep_df['aware_mrk']=='CU')]
        
        mrk_hep_df=mrk_hep_df[hep_df['difficulty']=='normal']
        
        drop_col=['trigger_stim', 'trigger_corr', 'trigger_aware','difficulty','cardiac_phase']
        mrk_hep_df.drop(drop_col,axis=1,inplace=True)
        
        
        
        # %%
        
        # ## Create markers for cardiac event before last one 
        # This time i don't want the cardiac event preceding, the reason is that given the short time of the systolic period all stimuli will fall within a few ms from the origin of the erp and the rest of the wave does not precede the stimulus 
        # 
        # These markers have the suffix 2 to indicate the -2 shift applyed to obtain them
        # 
        # This is an extremely exploratory analysis 
        # 
        # 
        # **important** : this means that the cardiac marker DOES NOT imply the phase in which the stimulus falls but the opposite
        
        # In[73]:
        
        
        hep2_df=merged_df_Tpk.loc[(merged_df_Tpk['cardiac_peak']!='no_peak')|(merged_df_Tpk['trigger_corr']!='no_stim')]
        hep2_df=hep2_df[(hep2_df['trigger_corr']!='no_stim').shift(-2).fillna(False)]
        hep2_df.reset_index(inplace=True)
        hep2_df['cardiac_peak']=hep2_df['cardiac_peak'].map({"R":"R2","T":"T2"})
        
        
        # In[74]:
        
        # In[64]:
        
        awa=['CA','CE','CU','UE','CA','CE','CU','UE']
        df_h=get_help_DF(hep2_df,['condition'],awa,'aware_mrk')
        
        
        
        # In[50]:
        
        
        hep2_df=hep2_df.merge(df_h,on=['condition'],how='left')
        
        
        # In[66]:
        on_col=['cardiac_peak','aware_mrk']
        
        awa_card=['T2UE','R2UE','T2CE','R2CE','T2CA','R2CA','T2CU','R2CU']
        
        awa_card.sort()
        
        df_h=get_help_DF(hep2_df,on_col,awa_card,'awareXcard')
        
        
        
        # In[67]:
        
        
        hep2_df=hep2_df.merge(df_h,on=on_col,how='left')
        
        hep2_df
        
        
        # In[68]:
        
        on_col=['awareXcard','rsp_phase']
        
        awa_card_rsp=['R2ECA','R2ICA','R2ECE','R2ICE','R2ECU','R2ICU','R2EUE','R2IUE','T2ECA','T2ICA','T2ECE','T2ICE','T2ECU','T2ICU','T2EUE','T2IUE']
        
        
        df_h=get_help_DF(hep2_df,on_col,awa_card_rsp,'awareXcardXrsp')
        
        
        # In[69]:
        
        
        hep2_df=hep2_df.merge(df_h,on=on_col,how='left')
        
        hep2_df
        
        
        # In[80]:
        
        type_sig='hep'
        
        file_end='hep2.feather'
        
        output_filename=files.out_filename(type_sig=type_sig,file_end=file_end)
        
        feather.write_dataframe(hep2_df, output_filename)
        
        
        # In[ ]:
        
        
        
        
        
        # In[81]:
        
        
        #save DF before this point but exclud easy stim for markers 
        mrk_hep2_df=hep2_df.loc[(hep2_df['aware_mrk']=='CA')|(hep2_df['aware_mrk']=='CU')]
        
        mrk_hep2_df=mrk_hep2_df[hep_df['difficulty']=='normal']
        drop_col=['trigger_stim', 'trigger_corr', 'trigger_aware','difficulty','cardiac_phase']
        mrk_hep2_df.drop(drop_col,axis=1,inplace=True)
        
        
        # In[ ]:
        
        
        
        
        
        # ## Create markers always for R peak preceding stimulus 
        # Name these markers RR 
        
        # In[82]:
        
        
        RR_df=merged_df.loc[(merged_df['cardiac_peak']=='R')|(merged_df['trigger_corr']!=0)]
        RR_df=RR_df[(RR_df['trigger_corr']!=0).shift(-1).fillna(False)]
        
        RR_df
        
        # In[83]:
        on_col=['condition']
        
        awa=['CA','CE','CU','UE','CA','CE','CU','UE']
        
        
        df_h=get_help_DF(RR_df,on_col,awa,'aware_mrk')
        
        
        
        # In[84]:
        
        
        RR_df=RR_df.merge(df_h,on=on_col,how='left')
        
        
        
        
        # In[83]:
        on_col=['condition']
        
        awa=['RRCA','RRCE','RRCU','RRUE','RRCA','RRCE','RRCU','RRUE']
        
        
        df_h=get_help_DF(RR_df,on_col,awa,'awareXcard')
        
        
        
        # In[84]:
        
        
        RR_df=RR_df.merge(df_h,on=on_col,how='left')
        
        
        
        
        # In[85]:
        on_col=['awareXcard','rsp_phase']
        
        awa_card_rsp=['RRECA','RRICA','RRECE','RRICE','RRECU','RRICU','RREUE','RRIUE']
        
        df_h=get_help_DF(RR_df,on_col,awa_card_rsp,'awareXcardXrsp')
        
        
        
        # In[86]:
        
        
        RR_df=RR_df.merge(df_h,on=on_col,how='left')
        
        RR_df
        
        
        # In[87]:
        
        
        
        type_sig='hep'
        file_end='hepRR.feather'
        
        output_filename=files.out_filename(type_sig=type_sig,file_end=file_end)
        
        feather.write_dataframe(RR_df, output_filename)
        
        
        # In[ ]:
        
        
        
        
        
        # In[88]:
        
        
        # save before this mrk 
        mrk_RR_df=RR_df.loc[(RR_df['aware_mrk']=='CA')|(RR_df['aware_mrk']=='CU')]
        mrk_RR_df=mrk_RR_df[RR_df['difficulty']=='normal']
        drop_col=['trigger_stim', 'trigger_corr', 'trigger_aware','difficulty','cardiac_phase']
        mrk_RR_df.drop(drop_col,axis=1,inplace=True)
        
        # In[ ]:
        
        
        
        
        
        # In[ ]:
        
        
        
        
        
        # # HRV & RSA 
        # ## HRV
        # recompure ecg signals using artefact correction so that heartbeats are normalized 
        
        # In[89]:
        
        
        ecg_signals_hrv, info_hrv = nk.ecg_process(ecg_sig, sampling_rate=srate)
        
        HRV=nk.hrv(info_hrv,sampling_rate=srate,show=True)
        
        
        # In[90]:
        
        
        HRV_fig= plt.gcf()
        
        
        # In[91]:
        
        
        HRV
        
        
        # In[92]:
        r_peaks_hrv=ecg_signals_hrv[ecg_signals_hrv['ECG_R_Peaks']==1]
        r_peaks_hrv.reset_index(inplace=True)
        r_peaks_hrv=r_peaks_hrv['index']
        
        
        HRV_lomb=nk.hrv_frequency(r_peaks_hrv,sampling_rate=srate,show=True,psd_method='lomb')
        
        
        # In[93]:
        
        
        HRV_lomb
        
        
        # In[94]:
        
        
        
        
        type_sig='phy_sig'
        file_end='HRV.feather'
        
        output_filename=files.out_filename(type_sig=type_sig,file_end=file_end)
        feather.write_dataframe(HRV, output_filename)
        
        
        file_end='HRV_lomb.feather'
        
        output_filename=files.out_filename(type_sig=type_sig,file_end=file_end)
        feather.write_dataframe(HRV_lomb, output_filename)
        
        
        type_sig='png'
        file_end='hrv_fig.png'
        
        output_filename=files.out_filename(type_sig=type_sig,file_end=file_end)
        HRV_fig.savefig(output_filename,dpi=2000)
        
        
        plt.close('all')
        
        
        # In[ ]:
        
        
        
        
        
        # ## RSA parametres
        
        # In[95]:
        
        
        rsa=nk.hrv_rsa(ecg_signals,rsp_signals=rsp_signals,rpeaks=info,sampling_rate=srate,continuous=False)
        
        
        
        # In[97]:
        
        
        rsa_df=pd.DataFrame.from_dict(rsa,orient='index').T
        
        
        # In[98]:
        
        
        
        type_sig='phy_sig'
        file_end='RSA.feather'
        
        output_filename=files.out_filename(type_sig=type_sig,file_end=file_end)
        
        feather.write_dataframe(rsa_df, output_filename)
        
        
        
        # In[99]:
        
        
        rsp_rate=rsp_signals[rsp_signals['RSP_Peaks']==1]
        rsp_rate=rsp_rate.RSP_Rate
        rsp_rate.describe()
        
        
        # In[100]:
        
        
        rsp_rate=nk.signal_rate(rsp_signals)
        
        rsp_peaks=nk.rsp_peaks(rsp_signals)
        
        
        
        # In[101]:
        
        
        rrv = nk.rsp_rrv( rsp_rate=rsp_rate,peaks=rsp_peaks, sampling_rate=srate, show=True,silent=False)
        rrv
        
        
        # # **Export**
        # ## write to [Feather](https://blog.rstudio.com/2016/03/29/feather/)
        # '''
        # 
        # import feather
        # 
        # path = 'my_data.feather'
        # 
        # feather.write_dataframe(df, path)
        # 
        # df = feather.read_dataframe('')
        # 
        # ''''
        
        # In[ ]:
        #Merge hep df 
        heps_df=pd.concat([mrk_hep_df,mrk_hep2_df])
        heps_df.sort_values(by='TF',inplace=True)    
        
        heps_df=heps_df.merge(mrk_RR_df,on=['TF','RT','condition','aware_mrk','rsp_phase'],how='outer',suffixes=["_hep","_RR"])
        
        heps_df.sort_values(by=['TF'],inplace=True)
        heps_df.cardiac_peak_hep.fillna('RR',inplace=True)
        
        col_fill=['awareXcard_hep', 'awareXcardXrsp_hep', 'cardiac_peak_RR',
               'awareXcard_RR', 'awareXcardXrsp_RR']
        for col in col_fill:
            heps_df[col]=heps_df[col].fillna('no_val')
        heps_df.drop(['cardiac_peak_RR'],axis=1,inplace=True)
        
        heps_df['ana_type']='hep'
        heps_df.rename(columns={"cardiac_peak_hep":"cardiac_phase"},inplace=True)    
        #%%
        #Merge hep_df + stim df
        
        mrk_df['ana_type']='vep'
        
        ans_df['ana_type']='ans'
        
        hep_stim_df=pd.concat([heps_df,mrk_df,ans_df])
        
        hep_stim_df.sort_values(by='TF',inplace=True)  
        
        for col in hep_stim_df.columns:
            hep_stim_df[col]= hep_stim_df[col].fillna('no_val') 
        
        
        #%%
        type_sig='stim'
        file_end='stim_hep.feather'
        
        output_filename=files.out_filename(type_sig=type_sig,file_end=file_end)
        
        feather.write_dataframe(hep_stim_df, output_filename)
        
        
        
        # In[102]:
        
        
        df_list=[mrk_df.columns.tolist()] + mrk_df.values.tolist()
        
        
        df_list
        
        
        # In[103]:
        
        
        df_hep_list=[mrk_hep_df.columns.tolist()] + mrk_hep_df.values.tolist()
        df_hep_list
        
        
        # In[104]:
        
        
        df_hep2_list=[mrk_hep2_df.columns.tolist()] + mrk_hep2_df.values.tolist()
        df_hep2_list
        
        
        # In[105]:
        
        
        df_RR_list=[mrk_RR_df.columns.tolist()] + mrk_RR_df.values.tolist()
        df_RR_list
        
        # %%
        df_heps_list=[heps_df.columns.tolist()] + heps_df.values.tolist()
        # %%
        
        df_ans_list=[ans_df.columns.tolist()] + ans_df.values.tolist()
        
        # In[106]:
        
        type_sig='phy_sig'
        file_end='.bdf.mrk'
        
        output_filename=files.out_filename(type_sig=type_sig,file_end=file_end,loc_folder='raw')
        
        
        with open(output_filename,'w') as output:
            output.write('TL02\n')
            for line in df_list[1:]:
                output.write(f'{line[0]}\t{line[0]}\t"{line[5]}"\n')
                output.write(f'{line[0]}\t{line[0]}\t"{line[6]}"\n')
                output.write(f'{line[0]}\t{line[0]}\t"{line[7]}"\n')
                output.write(f'{line[0]}\t{line[0]}\t"{line[8]}"\n')
            for line_hep in df_hep_list[1:]:
                output.write(f'{line_hep[0]}\t{line_hep[0]}\t"{line_hep[6]}"\n')
                output.write(f'{line_hep[0]}\t{line_hep[0]}\t"{line_hep[7]}"\n')
            for line2_hep in df_hep2_list[1:]:
                output.write(f'{line2_hep[0]}\t{line2_hep[0]}\t"{line2_hep[6]}"\n')
                output.write(f'{line2_hep[0]}\t{line2_hep[0]}\t"{line2_hep[7]}"\n')
                
            for line_RR in df_RR_list[1:]:
                output.write(f'{line_RR[0]}\t{line_RR[0]}\t"{line_RR[6]}"\n')
                output.write(f'{line_RR[0]}\t{line_RR[0]}\t"{line_RR[7]}"\n')
            for line_ans in df_ans_list[1:]:
                output.write(f'{line_ans[0]}\t{line_ans[0]}\t"{line_ans[7]}"\n')
                output.write(f'{line_ans[0]}\t{line_ans[0]}\t"{line_ans[8]}"\n')
                output.write(f'{line_ans[0]}\t{line_ans[0]}\t"{line_ans[9]}"\n') 

        
        
        # ## Create events in MNE from df
        # 
        # You can directly put events from annotation and save it into the mne df
        
        # In[107]:
        
        
        start=[]
        
        description=[]
        
        for line in df_list[1:]:
            start.append(line[0]/srate)
            description.append('vep/'+line[5]+'/'+line[3]+'/'+line[4])
        
        for line_hep in df_heps_list[1:]:
            start.append(line_hep[0]/srate)
            description.append('hep/'+line_hep[4]+'/'+line_hep[3]+'/'+line_hep[5]+'/'+line_hep[8])
        
        for line_ans in df_ans_list[1:]:
            start.append(line_ans[0]/srate)
            description.append('ans/'+line_ans[3]+'/'+line_ans[4])
        
        len_zero= len(description)
        
        duration=np.zeros(len_zero)
        
        
        event_annot=mne.Annotations(start,duration,description) 
        #events = mne.events_from_annotations(raw,event_annot)
        raw.set_annotations(event_annot)
        print(event_annot)
        
        
        # In[ ]:
        
        
        
        
        
        # In[ ]:
    
        
        # In[111]:
        
        
        #iir_params=dict(order=8,ftype='butter')
        eeg_filt=raw.copy().filter(0.5, 40 , method='fir',n_jobs='cuda')
        
        
        
        # In[112]:
        
        
        fig = eeg_filt.plot_psd(fmax=100, average=True,proj=False)
        
        
        # In[113]:
        
        
        eeg_ds=eeg_filt.resample(sfreq=256,n_jobs='cuda')
        
        
        # In[114]:
        
        
        eeg_ds.plot_psd(fmax=100, average=False,proj=True)
        
        
        # In[115]:
        
        
        eeg_ds.set_annotations(event_annot)
        
        print(event_annot)
        
        
        # In[116]:
        
        
        events_from_annot, event_dict = mne.events_from_annotations(eeg_ds)
        
        
        # In[117]:
        
        
        print(event_dict)
        print(events_from_annot[:5])
        
        
        # In[119]:
        
        
        eeg_ds.plot(events=events, start=5, duration=10,proj=True)
        
        
        # In[120]:
        type_sig='phy_sig'
        file_end='_ds_eeg.fif'
        
        output_filename=files.out_filename(type_sig=type_sig,file_end=file_end,loc_folder='raw')
        
        eeg_ds.save(output_filename,overwrite=True)
    
    
    # In[121]:
 # # HRV & RSA 
 # ## HRV
 # recompure ecg signals using artefact correction so that heartbeats are normalized

    ecg_sig=np.concatenate(ecg_sig_list,axis=1)
    ecg_sig=ecg_sig.flatten()

    
        
        
        
    resp=np.concatenate(rsp_sig_list,axis=1)
    
    resp=resp.flatten()
    
        
        
        
    rsp_signals,rsp_info=nk.rsp_process(resp,sampling_rate=srate)
    fig_rsp=nk.rsp_plot(rsp_signals,sampling_rate=srate)

    



       
 # In[89]:
        
        
    ecg_signals_hrv, info_hrv = nk.ecg_process(ecg_sig, sampling_rate=srate)
    ecg_fig = nk.ecg_plot(ecg_signals_hrv, sampling_rate=srate)
    
    
    type_sig='png'
    file_end='ecg_fig.png'
    output_filename=files.out_filename(type_sig=type_sig,file_end=file_end,short=True)
        
    ecg_fig.savefig(output_filename,dpi=2000)
        
    HRV=nk.hrv(info_hrv,sampling_rate=srate,show=True)
        
        
        # In[90]:
        
        
    HRV_fig= plt.gcf()
    
        
        # In[92]:
    r_peaks_hrv=ecg_signals_hrv[ecg_signals_hrv['ECG_R_Peaks']==1]
    r_peaks_hrv.reset_index(inplace=True)
    r_peaks_hrv=r_peaks_hrv['index']
        
        
    HRV_lomb=nk.hrv_frequency(r_peaks_hrv,sampling_rate=srate,show=True,psd_method='lomb')
        

        
    type_sig='phy_sig'
    file_end='HRV.feather'
        
    output_filename=files.out_filename(type_sig=type_sig,file_end=file_end,short=True)
    feather.write_dataframe(HRV, output_filename)
        
        
    file_end='HRV_lomb.feather'
        
    output_filename=files.out_filename(type_sig=type_sig,file_end=file_end,short=True)
    feather.write_dataframe(HRV_lomb, output_filename)
        
        
    type_sig='png'
    file_end='hrv_fig.png'
        
    output_filename=files.out_filename(type_sig=type_sig,file_end=file_end,short=True)
    HRV_fig.savefig(output_filename,dpi=2000)
        
        
    # In[ ]:
# ## RSA parametres

        
    rsa=nk.hrv_rsa(ecg_signals,rsp_signals=rsp_signals,rpeaks=info,sampling_rate=srate,continuous=False)
        
        
        # In[96]:
        
        
    rsp_signals[rsp_signals['RSP_Peaks']==1]
        
        
        # In[97]:
        
        
    rsa_df=pd.DataFrame.from_dict(rsa,orient='index').T
        
        
        # In[98]:
        
    type_sig='png'
    file_end='rsp_fig.png'
        
    output_filename=files.out_filename(type_sig=type_sig,file_end=file_end,short=True)    
        
        
    fig_rsp.savefig(output_filename,dpi=2000)
             
        
    type_sig='phy_sig'
    file_end='RSA.feather'
        
    output_filename=files.out_filename(type_sig=type_sig,file_end=file_end,short=True)
        
    feather.write_dataframe(rsa_df, output_filename)
        
        
        
        # In[99]:
        
        
    rsp_rate=rsp_signals[rsp_signals['RSP_Peaks']==1]
    rsp_rate=rsp_rate.RSP_Rate
    rsp_rate.describe()
        
        
        # In[100]:
        
        
    rsp_rate=nk.signal_rate(rsp_signals)
        
    rsp_peaks=nk.rsp_peaks(rsp_signals)
        
        
        
        # In[101]:
        
        
    rrv = nk.rsp_rrv( rsp_rate=rsp_rate,peaks=rsp_peaks, sampling_rate=srate, show=True,silent=False)
 
    rrv_fig= plt.gcf()
            
    type_sig='phy_sig'
    file_end='rrv.feather'
        
    output_filename=files.out_filename(type_sig=type_sig,file_end=file_end,short=True)
        
    feather.write_dataframe(rrv, output_filename)
    
    type_sig='png'
    file_end='rrv_fig.png'
        
    output_filename=files.out_filename(type_sig=type_sig,file_end=file_end,short=True)
    rrv_fig.savefig(output_filename,dpi=2000)
    
    plt.close('all')       
    
    
    # In[ ]:
    
    
    # In[ ]:
    
    
    #eog_epochs=mne.preprocessing.create_eog_epochs(raw,ch_name='C16',baseline=(-0.5,-0.2))
    
    
    # In[ ]:
    
    
    #eog_epochs.plot_image(combine='mean',vmin=-100,vmax=100)
    
    
    # In[ ]:
    
    
    #eog_epochs.average().plot_joint()
    
    
    # In[ ]:
    
    
    
    
