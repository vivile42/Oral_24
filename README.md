Code accompanying Leupin and Britz, in prep.
We here provide the code for preprocessing, epoching, and analysis of EEG data in sensor space.  EEG data can be made available upon request.
# Code organization
Each folder is generally organized with a main, helper and constants script.The base folder contains some helper functions to filter through the data directories.
- The main scripts contain the code that must be run.
- The helper scripts contain the helper functions and classes used to run the code.
- The constant files contain constants that are called in the script.
## Preprocessing
### Order to run preprocessing
1) markers/markers_main.py 
2) epochs/epochs_main.py
3) ICA
4) evoked/autoreject_main.py
5) evoked/evoked_MNE_main.py

### Description
1)	markers: analyzes cardiac and respiratory signals and generates markers to classify each stimulus according to the behavioral response and the cardiac / respiratory phase.
2)	epochs: segments the EEG into epochs before artifact rejection and computes ICA solutions.
3)	ICA: jupyter notebook to be applied to each subject to manually select ICA components to be rejected.
4)	evoked:
  - autoreject: used the Autoreject procedure to clean the epoched data after the ICA.
  - evoked_MNE: computes evoked potentials for each condition and each subject.
  - evoked_cleanbad: used to interpolate bad electrodes after inspection (dict with all electrodes is stored in evoked_constants)

## Analyses
The stats folder contains the code for statistical analyses.
### Behavioral
1) Behav_stats.ipynb: jupyter notebook to compute descriptive statistics and code to generate figures.
2) parametres_sub_o.ipynb: reads and compute descriptive parametres for subjects characteristics 
### Sensor space
1)	ERP_stats.ipynb: jupyter notebook that contains all analysis and figure output for the ERP analyses for awareness and cardiac/ resp phase.
2) Mass_anovas.ipynb: jupyter notebook that contains mass univariate anovas  FDR corrected 
3) Stats_helper contains helper functions used in sensor_space_stats.ipynb.



