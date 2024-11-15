#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 10:19:59 2021

@author: leupinv
"""
# functions that need to be revised to run TFCE and cluster analyses
from mne.channels import find_ch_adjacency
from mne.stats import spatio_temporal_cluster_1samp_test
from mne.stats import permutation_cluster_1samp_test
import mne
import numpy as np
import matplotlib.pyplot as plt


def get_DF_TFCE(evoked,crop_t=False,crop_value=None):
    if crop_t==True:
        data_crop=evoked[0].crop(crop_value[0],crop_value[1])
        data_shape=data_crop.data.shape
        subj_len=len(evoked)
        print(data_shape)
    else:
        data_shape=evoked[0].data.shape
        subj_len=len(evoked)

    X=np.empty((subj_len,data_shape[1],data_shape[0]))
    for idx, ev in enumerate(evoked):
        X[idx,:,:]=ev.crop(crop_value[0],crop_value[1]).data.T
    print(X.shape)
    return X


def get_Ttest_TFCE(X,plot_times='peaks',adjacency=None,averages=None,permutations=None):
    tfce = dict(start=0, step=.05)

    t_obs, clusters, cluster_pv, h0 = permutation_cluster_1samp_test(
    X, tfce, adjacency=adjacency,
    n_permutations=permutations,out_type='mask',n_jobs=-1)  # a more standard number would be 1000+

    significant_points = cluster_pv.reshape(t_obs.shape).T < .05
    print(str(significant_points.sum()) + " points selected by TFCE ...")
    biosemi_montage = mne.channels.make_standard_montage('biosemi128')
    n_channels = len(biosemi_montage.ch_names)
    info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=256.,
                                ch_types='eeg')
    evok=mne.EvokedArray(t_obs.T,info,tmin=-0.1)
    evok.set_montage(biosemi_montage)
    evok.plot_image(mask=significant_points,scalings=1,units='T-value',show_names='auto')

    evok.plot_topomap(plot_times,outlines='head',scalings=1,units='T-value',average=averages,mask=significant_points)
    return t_obs, clusters, cluster_pv, h0

def tTest_TFCE_ana(evoked,crop_t=False,crop_value=None,FDR=False,plot_times='peaks',averages=None,permutations=1000,TFCE=False):
    adjacency, _ = find_ch_adjacency(evoked[0][0].info, "eeg")
    evoked_1=get_DF_TFCE(evoked[0],crop_t=True,crop_value=crop_value)
    evoked_2=get_DF_TFCE(evoked[1],crop_t=True,crop_value=crop_value)
    X=evoked_1-evoked_2
    if TFCE:
        t_obs, clusters, cluster_pv, h0=get_Ttest_TFCE(X,plot_times=plot_times,averages=averages,permutations=permutations)
        return t_obs, clusters, cluster_pv, h0
    else:
        t_obs, clusters, cluster_pv, h0=get_Ttest_cluster(X,plot_times=plot_times,averages=averages,permutations=permutations)
        return t_obs, clusters, cluster_pv, h0

def get_Ttest_cluster(X,plot_times='peaks',adjacency=None,averages=None,permutations=1000):

    t_obs, clusters, cluster_pv, h0 = permutation_cluster_1samp_test(
    X, adjacency=adjacency,
    n_permutations=permutations,out_type='mask')  # a more standard number would be 1000+

    T_obs_plot=np.nan*np.ones_like(t_obs)
    for c,p_val in zip(clusters,cluster_pv):
        if p_val<=.05:
            T_obs_plot[c]=t_obs[c]
    data=T_obs_plot.T
    data=np.flipud(data)

    plt.figure()
    plt.imshow(data,cmap=plt.cm.RdBu_r)
    plt.show()



    return t_obs, clusters, cluster_pv,T_obs_plot
