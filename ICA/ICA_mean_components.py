import base.files_in_out as files_in_out
import base.base_constants as b_cs
import os
import platform
import ast

platform.system()

# define starting datafolder

if platform.system() == 'Darwin':
    os.chdir('/Volumes/BBC/BBC/WP1/data/EEG/tsk')
    base_datafolder = '/Volumes/Elements/'
else:
    os.chdir('Z:/BBC/WP1/data/EEG/tsk/')

list_dicts=[]

for g in b_cs.G_N:
    files = files_in_out.GetFiles(
        filepath='preproc', eeg_format='ICA_log.txt', g_num=g)
    file_log=files.taskfiles[0]
    
    with open(file_log) as log:
        dict_el={}
        for line in log:
            key,value=line.split('=')
            key=key.replace(' ','_')
            value=ast.literal_eval(value)
            dict_el[key]=value
        n_eog=len(dict_el['eog_index'])

        n_mio=len(dict_el['artefact_index'])

        n_art=n_eog+n_mio

        
    list_dicts.append(n_art)

mean_art=sum(list_dicts)/(len(b_cs.G_N))
    
    
