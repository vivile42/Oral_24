#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 09:35:43 2021

@author: leupinv
"""
import markers.markers_constants as cs
import markers.markers_helper as hp

import base.files_in_out as files_in_out
import base.base_constants as b_cs
import markers.markers_MNE_helper as mne_hp
import matplotlib.pyplot as plt
import gc



for g_n in b_cs.G_N:
    for cond in cs.condition[0]:
        files = files_in_out.GetFiles(
            filepath=cs.base_datafolder, condition=cond, g_num=g_n)
        tskfiles = files.condition_files
        rsp_sig_list = []
        card_sig_list = []
        raw_list = []

        for idx in range(files.condition_nfiles):

            files.get_info(idx)
            mne_data = mne_hp.MarkersMNE(files)
            raw_list.append(mne_data.raw)



        mne_data.merge_raws(raw_list)
        mne_data.get_ds_eeg(mne_data.raws, open_file=False)
        mne_data.get_triggers()
        mne_data.get_card(raws=mne_data.raws)
        mne_data.get_rsp(raws=mne_data.raws)
        mne_data.merge_rsp_DF()
        mne_data.get_ecg_stim_DF()
        mne_data.get_ecg_hep_DF()

        DF_class=hp.DF_Markers(mne_data)
        annot=DF_class.get_annotations()
        DF_class.get_metadata()

        mne_data.update_annot(annot=annot,append=True)
        mne_data.get_HRV()
        rsa,rrv=mne_data.get_RSA()
        mne_data.save_df()

        mne_data.save_raws(raw_list)
        del([mne_data,DF_class,annot,raw_list,rsp_sig_list,card_sig_list])
        ##makes sure to clean cache
        gc.collect()
        #plt.close('all')

#%%

#%%
