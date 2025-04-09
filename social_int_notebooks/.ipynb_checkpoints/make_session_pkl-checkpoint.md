---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Make sess class for each session and save as pickle

Currently uses info from sessions_dict.py to loop through sessions and create the sess class.

sess pickle files will be named `<scene>_<session>_<scan>.pickle`  \
and saved in `path_dict['preprocessed_root']/sess/<animal>/<date>`.

Set `overwrite` to `True` if you want to overwrite existing .pickle files. Otherwise, you will get an error that the file already exists.

```python
overwrite = True
```

```python
import os
import glob
import pickle
import numpy as np

import pandas as pd
# import InVivoDA_analyses
import social_int_analyses

# from InVivoDA_analyses import preprocessing as pp
from InVivoDA_analyses import utilities as ut

from social_int_analyses import utilities_ES as u
from social_int_analyses import sleap_utils as slp


import TwoPUtils
from TwoPUtils import preprocessing as pp

from suite2p.io.binary import BinaryFile
from PIL import Image
import PIL

import matplotlib.pyplot as plt

%load_ext autoreload
%autoreload 2
```

### Specify your path dictionary here.

```python
from social_int_analyses.path_dict_esay import path_dictionary as path_dict
# options: path_dict_josquin, path_dict_msosamac
path_dict
```

```python
from social_int_analyses.social_int_sess_deets import social_2P_sessions
from social_int_analyses.social_int_sess_deets import social_mice
```

```python
mouse = social_mice[2]
d = social_2P_sessions[mouse][0]
print(d)

date, scene, session, scan = d['date'], d['scene'], d['session'], d['scan']
print(scene)

vrdir = path_dict['VR_Data']
basedir = os.path.join(path_dict['sbx_root'], mouse,date,scene)
stem =  os.path.join(basedir, f'{scene}_{session:03}_{scan:03}')
source_folder =  'C:/Users/esay/data/social_interaction/SLEAPData'

if 'diffsex' in scene:
    if 'unrestrict' in scene:
        basedir = os.path.join(path_dict['sbx_root'], mouse,date,'social_unrestrict_nov_diffgender')
        stem = os.path.join(basedir, 'social_unrestrict_nov_diffgender_'f'{session:03}_{scan:03}')
    else:
        basedir = os.path.join(path_dict['sbx_root'], mouse,date,'social_restrict_nov_diffgender')
        stem =os.path.join(basedir, 'social_restrict_nov_diffgender_'f'{session:03}_{scan:03}')

source_stem = os.path.join(source_folder, mouse, date, (scene +'.h5') )

d.update({'mouse': mouse ,
          
          'scan_file':stem + '.sbx',
          'scanheader_file': stem + '.mat',
          'vr_filename': os.path.join("C://Users/esay/data/social_interaction/VRData",mouse,date,"%s_%d.sqlite" %(scene,session)),
          'scan_number': scan,
          'prompt_for_keys': False,
          'VR_only': False,
          'scanner': "NLW",
          'n_channels':2,
          'n_planes':3
             })
source_stem
```

```python
sess = TwoPUtils.sess.Session(**d)
sess.load_scan_info(sbx_version=3) #check sess.scan_info
sess.align_VR_to_2P()
# depends on vr being loaded already
sess.align_SLEAP_to_2P()
# sess.tunnel_data.shape, sess.vr_data.shape
```

```python
TwoPUtils.sess.save_session(sess,'C:/Users/esay/data/social_interaction/SessPkls')
```

```python
interaction = sess.tunnel_data['interaction']
plt.figure(figsize=(10,5))
for i in range(len(sess.tunnel_data['interaction'])):
    if interaction[i] ==1:
        plt.axvspan(i, i+1, color = 'pink', alpha=0.1)
# plt.plot(sess.tunnel_data['head_velocity'])
plt.plot(sess.vr_data['dz'])
```

```python
def plot_dz_interaction(ax, sess):
    dz = sess.vr_data['dz']
    if isinstance(sess.tunnel_data, bool):
        interaction = None
    else:
        interaction = sess.tunnel_data['interaction']
        
    frames = np.arange(len(dz))
    ax.plot(frames, dz, label='dz',color='blue')
    # plt.figure(figsize=(10,5))
    # plt.plot(frames, dz, label='dz', color ='blue')

    if isinstance(sess.tunnel_data, bool):
        return 
    else:
        for i in range(len(interaction)):
            if interaction[i] ==1:
                ax.axvspan(i, i+1, color = 'pink', alpha=0.1)
    
    plt.xlabel("Frames")
    plt.ylabel("Running wheel dz")
    # plt.title("dz with interaction")
    # plt.legend()
    # plt.show()
```

```python
sess.vr_data
```

```python
sess.tunnel_data
```

```python
import seaborn as sns
```

```python
def plot_interaction_comparison(social_mice, social_2P_sessions):
    fam_interactions = []
    nov_interactions = []
    for mouse in social_mice:
        for day in range(len(social_2P_sessions[mouse])):
            d = social_2P_sessions[mouse][day]
            date, scene, session, scan = d['date'], d['scene'], d['session'], d['scan']
            pkldir = os.path.join('Z:/giocomo/esay/cd_project/social_interaction/SessPkls', mouse,  date, f'{scene}_{session}.pkl')
            print("Loading:", pkldir)
            with open(pkldir, 'rb') as file:
                sess = dill.load(file)
                
            if isinstance(sess.tunnel_data, bool):
                interaction = None
            else:
                
                num_interactions = np.sum(sess.tunnel_data['interaction'] == 1)
                if 'fam' in sess.scene:
                    fam_interactions.append(num_interactions)
                elif 'nov' in sess.scene:
                    nov_interactions.append(num_interactions)
                    
    data = {'Scene Type': ['fam'] * len(fam_interactions) + ['nov'] * len(nov_interactions),
            'Interactions': fam_interactions + nov_interactions}
    df = pd.DataFrame(data)
    plt.figure(figsize=(6, 6))
    colors = ['blue','purple']
    sns.barplot(x='Scene Type', y='Interactions', data=df,palette=colors)#, ci='sem', capsize=0.1, errcolor='black')
    sns.stripplot(x='Scene Type', y='Interactions', data=df, color='black', jitter=True, size=5)
    plt.xlabel("Scene Type")
    plt.ylabel("Number of Interaction Frames")
    plt.title("Comparison of Interaction Frames between Scene Types")
    plt.show()
```

```python
plot_interaction_comparison(social_mice, social_2P_sessions)
```

```python
def visualize_multiple_sessions(social_mice, social_2P_sessions):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))  # Adjust grid size as needed
    axes = axes.flatten()
    idx = 0
    for day in range(18):
        for mouse in social_mice[1:2]:
            if idx >=len(axes):
                break
            d = social_2P_sessions[mouse][day]
            date, scene, session, scan = d['date'], d['scene'], d['session'], d['scan']
            pkldir = os.path.join('Z:/giocomo/esay/cd_project/social_interaction/SessPkls', mouse,  date, f'{scene}_{session}.pkl')
            print("Loading:", pkldir)
            with open(pkldir, 'rb') as file:
                sess = dill.load(file)
            plot_dz_interaction(axes[idx],sess)
            axes[idx].set_title(f"{mouse},  {scene}")
            idx +=1
    plt.tight_layout()
    plt.show()
```

```python
visualize_multiple_sessions(social_mice, social_2P_sessions)
```

```python
import dill
mouse = social_mice[1]
d = social_2P_sessions[mouse][7]
date, scene, session, scan = d['date'], d['scene'], d['session'], d['scan']
# pkldir = os.path.join('Z:/giocomo/esay/cd_project/social_interaction/SessPkls', mouse,  date, f'{scene}_{session}.pkl')
pkldir = os.path.join('C:/Users/esay/data/social_interaction/SessPkls', mouse,  date, f'{scene}_{session}.pkl')

print(pkldir)

with open(pkldir, 'rb') as file:
    sess = dill.load(file)
```

```python
sess.s2p_path = os.path.join("Z:/giocomo/candong/social_interaction_data/calcium_imaging", sess.mouse, sess.date)
```

```python
sess.s2p_stats = os.path.join("Z:/giocomo/candong/social_interaction_data/calcium_imaging", sess.mouse, sess.date,"combined/suite2p/combined/stat.npy")
```

```python
sess.s2p_path
```

```python
sess.load_suite2p_data_multi_session(multi_sess=True)

```

```python
sess.scene
```

```python
TwoPUtils.sess.save_session(sess,'C:/Users/esay/data/social_interaction/SessPkls')
```

```python
sess.s2p_path
```

```python
sess.s2p_ops['data_path']
```

```python
data_path = ['/home/candong/oak/candong/social_interaction_data/calcium_imaging/social-0914-4/03_10_2024/Env1_to_Env2_fixreward',
 '/home/candong/oak/candong/social_interaction_data/calcium_imaging/social-0914-4/03_10_2024/social_restrict_nov',
 '/home/candong/oak/candong/social_interaction_data/calcium_imaging/social-0914-4/03_10_2024/social_unrestrict_nov',
 '/home/candong/oak/candong/social_interaction_data/calcium_imaging/social-0914-4/03_10_2024/social_restrict_fam',
 '/home/candong/oak/candong/social_interaction_data/calcium_imaging/social-0914-4/03_10_2024/social_unrestrict_fam',
 '/home/candong/oak/candong/social_interaction_data/calcium_imaging/social-0914-4/03_10_2024/social_wheel_restrict']
```

```python
ops = TwoPUtils.s2p.set_ops(d={'save_path0': 'Z:/giocomo/candong/social_interaction_data/calcium_imaging/social-0914-4\\04_10_2024\\Env1_to_Env2_fixreward\\Env1_to_Env2_fixreward_001_001',
                        'data_path': data_path,
                       'save_path0': fullpath,
                       'fast_disk':[],
                       'move_bin':True,
                       'two_step_registration':True,
                       'maxregshiftNR':10,
                       'nchannels':2,
                       'tau': 0.7,
                       'functional_chan':1,
                        'align_by_chan' : 1,
                       'nimg_init': 2000,
                       'fs':info['frame_rate'],
                       'roidetect':True,
                       'input_format':"h5", #h5
                       'h5py_key':'data',
                       'sparse_mode':True,
                       'threshold_scaling':.8, #.6
                        'sbx_ndeadcols': 100,
                        'nplanes':nplanes})
```

```python

```
