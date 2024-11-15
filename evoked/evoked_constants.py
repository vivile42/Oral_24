# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 15:24:53 2021

@author: Engi
"""

import os
import platform


platform.system()

# define starting datafolder

if platform.system() == 'Darwin':
    os.chdir('/Volumes/BBC/BBC/WP1/data/EEG/tsk')
    base_datafolder = '/Volumes/Elements/'
else:
    os.chdir('Z:/BBC/WP1/data/EEG/tsk/')
    #os.chdir('c:/Users/Engi/all/BBC/WP1/data/EEG/tsk')
    #base_datafolder = 'E:/'

eeg_format = 'fif'
eeg_exp = 'tsk'
datafolder = 'preproc'
condition = ['n','o']


sys_mask=['sys_mask==1', 'noh']
sys_lab = ['maskON', 'maskOFF']

#sys_lab=['sysEAR','sysLAT','maskNEG','maskON','inhEAR','inhLAT','exhEAR','exhLAT','maskOFF']



only_vep = ['sysEAR', 'sysLAT', 'inhEAR', 'inhLAT', 'exhEAR', 'exhLAT']

heart_cond = ['cfa','nc']

diffi_list = ['normal']

accuracy_cond = ['correct','mistake']

id_vep = ['aware', 'unaware', 'dia', 'sys', 'inh', 'exh', 'aware/dia', 'unaware/dia', 'aware/sys',
          'unaware/sys', 'aware/inh', 'unaware/inh', 'aware/exh', 'unaware/exh',
                         'aware/sys/inh', 'aware/sys/exh', 'aware/dia/inh', 'aware/dia/exh',
                         'unaware/sys/inh', 'unaware/sys/exh', 'unaware/dia/inh', 'unaware/dia/exh',
                         'sys/inh', 'sys/exh', 'dia/inh', 'dia/exh']

id_hep_type = ['R', 'R2', 'T', 'T2']

comb_type = ['aware', 'unaware', 'inh', 'exh']


id_hep = ['/'.join([x, y]) for x in id_hep_type for y in comb_type]

id_hep2 = ['/'.join([x, y, z]) for x in id_hep_type for y in comb_type[:2]
           for z in comb_type[-2:]]

id_hep3 = ['RRCA', 'RRCU']

id_hep_fin = id_hep_type+id_hep+id_hep2+id_hep3

print(id_hep)
id_hep = ['aware', 'unaware', 'dia', 'sys', 'inh', 'exh', 'aware/dia', 'CU/dia', 'aware/sys',
          'unaware/sys', 'aware/inh', 'unaware/inh', 'aware/exh', 'unaware/exh']


id_xns = ['aware', 'unaware', 'dia', 'sys', 'inh', 'exh', 'aware/dia', 'unaware/dia', 'aware/sys',
          'unaware/sys', 'aware/inh', 'unaware/inh', 'aware/exh', 'unaware/exh']


# Constants convert to eeglab
eeg_format_conv = 'vep_clean_epo.fif'

#constants for cleaning bad channels:

clean_dict_n = [['g01', dict(vep=None, hep=['C14', 'C28'])],
              ['g08', dict(vep=None, hep=['C27'])],
              ['g10', dict(vep=None, hep=['C5'])],
              ['g11', dict(vep=None, hep=['C5'])],
              ['g12', dict(vep=['A1'], hep=['A1'])],
              ['g16', dict(vep=['A1'], hep=['A1'])],
              ['g24', dict(vep=None, hep=['A17'])],
              ['g28', dict(vep=None, hep=['C27', 'C28'])],
              ['g33', dict(vep=None, hep=['A22', 'C31', 'D21'])],
              ['g37', dict(vep=['A1'], hep=['D1'])],
              ['g40', dict(vep=None, hep=['A4', 'C4', 'C19'])],
              ['g41', dict(vep=None, hep=['C1', 'C21'])],
              ['g42', dict(vep=None, hep=['B25', 'B27'])],
              ['g45', dict(vep=['D1', 'D3'], hep=['D1', 'D3'])],
              ['g46', dict(vep=None, hep=['A11', 'C28'])],
              ['g47', dict(vep=None, hep=['C15', 'C28', 'D12', 'D13'])],
              ['g49', dict(vep=None, hep=['A2', 'C14', 'C21'])],
              ]
