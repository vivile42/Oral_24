

from base.files_in_out import GetFiles, filter_list

import evoked.evoked_constants as cs
import base.base_constants as b_cs

from evoked.evoked_MNE_helper import EpochGroup
import gc
import mne

# In[Get list of epochs]
import os
import platform


platform.system()

# define starting datafolder

if platform.system() == 'Darwin':
    os.chdir('/Volumes/BBC/BBC/WP1/data/EEG/tsk')
    base_datafolder = '/Volumes/Elements/'
else:
    os.chdir('Z:/BBC/WP1/data/EEG/tsk/')
    
sys_lab = cs.sys_lab

for idx, sys in enumerate(sys_lab):
    for cfa in cs.heart_cond:
        epochs_group = EpochGroup()
        for g_n in b_cs.G_N:

            for cond in cs.condition[1]:
                files = GetFiles(filepath=cs.datafolder,
                                 condition=cond, g_num=g_n,
                                 eeg_format='clean_epo.fif')

            for file in files.condition_files:
                if cfa in file:
                    if 'vep' in file:
                        epochs_group.add_epoch(file, g_num=g_n, idx=idx)

        epochs_group.get_all_list()
        epochs_group.get_averages(cfa,miss=False)
        del epochs_group
        gc.collect()


# # In[Get list of epochs]


# evoked_CA=[epo['normal/RRCA'].average() for epo in epochs_group.hep_group]

# grand_average_CA=mne.grand_average(evoked_CA)


# grand_average_CA.plot_joint()

# # In[Get list of epochs]

# evoked_CU=[epo['normal/RRCU'].average() for epo in epochs_group.hep_group]

# grand_average_CU=mne.grand_average(evoked_CU)


# grand_average_CU.plot_joint()

#     #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Thu Jul 22 13:49:41 2021

# @author: leupinv
# """


# for g_n in b_cs.G_N:
#     for cond in cs.condition:
#         files = GetFiles(filepath=cs.datafolder,
#                                       condition=cond,g_num=g_n,
#                                       eeg_format=cs.eeg_format)

#         files.condition_files=filter_list(files.condition_files,'_clean_')
#         print(files.condition_files)
#         n=0
        # for file in files.condition_files:

        #     files.get_info(index=n,end_fix=-14,start_fix=27)
        #     filename=files.current_file_dir
        #     auto_rej=auto_hp.AutoHelp(file)
        #     auto_rej.get_erps_MNE(files)

        #     n+=1

#%%

#%%
