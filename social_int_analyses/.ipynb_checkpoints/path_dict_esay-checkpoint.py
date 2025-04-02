'''
This is an example path dictionary file. 
Edit the ROOT paths for you specific system 
     and save as "path_dict_user" or "path_dict_machine"
     for easy future access.

Example usage in a jupyter notebook:
` from InVivoDA_analyses.path_dict_msosa import path_dictionary as path_dict `

'''

import os

### REMOTE ###
# RCLONE_DATA_ROOT = "DATA"

###  LOCAL  ###
HOME = os.path.expanduser("~")

DATA_ROOT = os.path.join("Z:/giocomo/candong/social_interaction_data/calcium_imaging")  # parent path to data
PP_ROOT = ('C:/Users/esay/data/social_interaction') # path to preprocessed data
SBX_ROOT = ("Z:/giocomo/candong/social_interaction_data/calcium_imaging") #os.path.join("/Volumes") # scanbox data path, if different from preprocessed data path

GIT_ROOT = os.path.join(HOME,"repos")

FIG_DIR = os.path.join(DATA_ROOT,"fig_scratch")
#os.path.join("/Users/marielenasosa/Library/Mobile Documents/com~apple~CloudDocs","Data","fig_scratch")


path_dictionary = {
    "preprocessed_root": PP_ROOT,
    "sbx_root": SBX_ROOT,
    "VR_Data": os.path.join(PP_ROOT,"VRData"),
    "git_repo_root": GIT_ROOT,
    "TwoPUtils": os.path.join(GIT_ROOT,"TwoPUtils"),
    "home": HOME,
    "fig_dir": FIG_DIR,
}


os.makedirs(path_dictionary['fig_dir'],exist_ok=True)