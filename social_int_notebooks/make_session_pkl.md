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

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### import dependencies
<!-- #endregion -->

```python
overwrite = True
```

```python
import os
import glob
import pickle
import numpy as np

import pandas as pd
import social_int_analyses

from social_int_analyses import utilities_ES as u
from social_int_analyses import sleap_utils as slp
from social_int_analyses import alignment

import TwoPUtils
from TwoPUtils import preprocessing as pp

from suite2p.io.binary import BinaryFile
from PIL import Image
import PIL
import dill
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
from social_int_analyses.social_int_sess_deets import social_VR_sessions
from social_int_analyses.social_int_sess_deets import social_mice

```

```python
#NEED TO MOVE THESE TO UTILS EVENTUALLY 

def update_sess_dict(mouse, day, KO = True):
    d = social_VR_sessions[mouse][day]
    date, scene, session, scan = d['date'], d['scene'], d['session'], d['scan']
    print(scene)
    
    vrdir = path_dict['VR_Data']
    basedir = os.path.join(path_dict['sbx_root'], mouse,date,scene)
    stem =  os.path.join(basedir, f'{scene}_{session:03}_{scan:03}')
    source_folder =  'C:/Users/esay/data/social_interaction/SLEAP_raw/videos' # CHANGE SO DEPENDENT ON PATH DICT
    source_stem = os.path.join(source_folder, mouse, (scene +'.h5') )
    
    d.update({'mouse': mouse ,
              
              'scan_file':stem + '.sbx',
              'scanheader_file': stem + '.mat',
              'vr_filename': os.path.join("C://Users/esay/data/social_interaction/VRData",mouse,date,"%s_%d.sqlite" %(scene,session)),  # CHANGE SO DEPENDENT ON PATH DICT
              'scan_number': scan,
              'prompt_for_keys': False,
              'VR_only': False,
              'scanner': "NLW",
              'n_channels':1,
              'n_planes':3
                 })
    return d

def run_and_save(d):
    sess = TwoPUtils.sess.Session(**d)
    sess.load_scan_info(sbx_version=3) #check sess.scan_info
    alignment.align_VR_to_2P(sess)
    
    # depends on vr being loaded already
    alignment.align_SLEAP_to_2P(sess)
    TwoPUtils.sess.save_session(sess,'C:/Users/esay/data/social_interaction/SessPkls')  # CHANGE SO DEPENDENT ON PATH DICT
    
# source_stem
```

```python
social_mice

```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### for loop to create sess files for all mice
<!-- #endregion -->

```python
for mouse in social_mice:
    print(mouse)
    for day in range(18):
        print(day)
        d = update_sess_dict(mouse, day)
        run_and_save(d)

```

### try generating one sess file first

```python
mouse = 'social-0057-1'
day = 0
d = update_sess_dict(mouse, day)

sess = TwoPUtils.sess.Session(**d)
sess.load_scan_info(sbx_version=3) #check sess.scan_info
alignment.align_VR_to_2P(sess)
# depends on vr being loaded already
alignment.align_SLEAP_to_2P(sess)
# sess.tunnel_data.shape, sess.vr_data.shape
```

```python
TwoPUtils.sess.save_session(sess,'C:/Users/esay/data/social_interaction/SessPkls')
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### everything below this is just me messing with data
<!-- #endregion -->

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
    diffsex_interactions = []
    for mouse in social_mice:
        for day in range(len(social_2P_sessions[mouse])):
            d = social_2P_sessions[mouse][day]
            date, scene, session, scan = d['date'], d['scene'], d['session'], d['scan']
            # pkldir = os.path.join('Z:/giocomo/esay/cd_project/social_interaction/SessPkls', mouse,  date, f'{scene}_{session}.pkl')
            pkldir = os.path.join('C:/Users/esay/data/social_interaction/SessPkls', mouse,  date, f'{scene}_{session}.pkl')
            print("Loading:", pkldir)
            with open(pkldir, 'rb') as file:
                sess = dill.load(file)
                
            if isinstance(sess.tunnel_data, bool):
                interaction = None
            else:
                
                num_interactions = np.sum(sess.tunnel_data['interaction'] == 1)
                total_frames = len(sess.tunnel_data['interaction'])
                perc_int = (num_interactions / total_frames)*100
                if 'sex' in sess.scene:
                    diffsex_interactions.append(perc_int)
                elif 'fam' in sess.scene:
                    fam_interactions.append(perc_int)
                elif 'nov' in sess.scene and 'nov_diffsex' not in sess.scene:
                    nov_interactions.append(perc_int)
                    
    data = {'Scene Type': ['fam'] * len(fam_interactions) + ['nov'] * len(nov_interactions) + ['diffsex'] * len(diffsex_interactions),
            'Interactions': fam_interactions + nov_interactions + diffsex_interactions}
    df = pd.DataFrame(data)
    plt.figure(figsize=(6, 6))
    colors = ['blue','purple','pink']
    sns.barplot(x='Scene Type', y='Interactions', data=df,palette=colors)#, ci='sem', capsize=0.1, errcolor='black')
    sns.stripplot(x='Scene Type', y='Interactions', data=df, color='black', jitter=True, size=5)
    plt.xlabel("Scene Type")
    plt.ylabel("Interaction (%)")
    # plt.title("Comparison of Interaction Frames between Scene Types")
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
d = social_2P_sessions[mouse][6]
date, scene, session, scan = d['date'], d['scene'], d['session'], d['scan']
# pkldir = os.path.join('Z:/giocomo/esay/cd_project/social_interaction/SessPkls', mouse,  date, f'{scene}_{session}.pkl')
pkldir = os.path.join('C:/Users/esay/data/social_interaction/SessPkls', mouse,  date, f'{scene}_{session}.pkl')

print(pkldir)

with open(pkldir, 'rb') as file:
    sess = dill.load(file)
```

```python
sess.vr_data
```

```python
interaction_idx = np.where(sess.tunnel_data['interaction']==1)[0]

if len(interaction_idx)==0:
    sess.trial_matrices['int_start'] = np.array([])
    sess.trial_matrices['int_end'] = np.array([])
else:
    breaks = np.where(np.diff(interaction_idx) > 5)[0]

    bout_starts = interaction_idx[np.insert(breaks + 1, 0, 0)]
    bout_ends = interaction_idx[np.append(breaks , len(interaction_idx) -1)]

    sess.trial_matrices['int_start'] = bout_starts
    sess.trial_matrices['int_end'] = bout_ends
```

```python
sess.trial_matrices
```

```python
def speed_vs_interaction(mice, sessions, window =100):
    all_speed = []  # Will hold [pre, during_avg, post] per bout, across all sessions
    for mouse in mice:
        for d in sessions[mouse]:

            if 'wheel' in d['scene']:
                continue
            session_traces = []
            date, scene, session, scan = d['date'], d['scene'], d['session'], d['scan']
            # pkldir = os.path.join('Z:/giocomo/esay/cd_project/social_interaction/SessPkls', mouse,  date, f'{scene}_{session}.pkl')
            pkldir = os.path.join('C:/Users/esay/data/social_interaction/SessPkls', mouse,  date, f'{scene}_{session}.pkl')
    
            print(pkldir)
            try: 
                with open(pkldir, 'rb') as file:
                    sess = dill.load(file)
            except FileNotFoundError:
                print(f"File not found:{pkldir}")
                continue
                
            if isinstance(sess.tunnel_data, bool) or 'interaction' not in sess.tunnel_data:
                continue
                
            interaction_indices = np.where(sess.tunnel_data['interaction'] == 1)[0]
    
            if len(interaction_indices) == 0:
                continue
    
            breaks = np.where(np.diff(interaction_indices) > 1)[0]
            int_starts = interaction_indices[np.insert(breaks + 1, 0, 0)]
            # bout_ends = interaction_indices[np.append(breaks, len(interaction_indices) - 1)]
    
            filter_starts = []
            last_accepted = -np.inf
            for s in int_starts:
                if s - last_accepted >=window:
                    filter_starts.append(s)
                    last_accepted = s
    
            sess.trial_matrices['int_start'] = np.array(filter_starts)
            # sess.trial_matrices['int_end'] = bout_ends
    
            
            speed = sess.vr_data['speed']
    
            for start in filter_starts:
                if start < window or start + window >=len(speed):
                    continue
    
                trace = speed[start-window: start +window+1]
                if np.all(trace <=20):
                    continue
                    
                session_traces.append(trace)

            if session_traces:
                all_speed.append(np.array(session_traces))

    return all_speed

```

```python
len(speed_traces)
```

```python
sess.vr_data['speed'][266]
```

```python
mouse = social_mice[0]
mouse
```

```python
speed_traces = speed_vs_interaction(social_mice, social_2P_sessions)
```

```python
def plot_speed_vs_interaction(speed_traces, window = 100):
    if len(speed_traces) == 0:
        print("No speed traces to plot.")
        return

    sess_means = [np.nanmean(traces, axis = 0) for traces in speed_traces if len(traces) > 0]
    sess_means = np.array(sess_means)

    grand_mean = np.nanmean(sess_means, axis=0)
    sem = np.nanstd(sess_means, axis=0) / np.sqrt(sess_means.shape[0])

    x = np.arange(-window, window+1)

    plt.figure(figsize=(10,4))
    
    # for sm in sess_means:
    #     plt.plot(x, sm, color='black', alpha=0.2, linewidth=0.8)
        
    
    plt.plot(x, grand_mean, color='red', label='Mean')
    plt.fill_between(x, grand_mean - sem, grand_mean + sem, color='red', alpha=0.1)
    plt.axvline(0, color='blue', linestyle='--', label='interaction')
    plt.xlabel('Frames relative to interaction')
    # plt.ylim(20,50)
    plt.ylabel('Running wheel speed')
    plt.title('Running wheel speed during interaction bout')
    # plt.legend()
    plt.tight_layout()
    plt.show()
```

```python
plot_speed_vs_interaction(speed_traces)
```

```python
def plot_avg_speed_with_sem(all_speed_traces, pre_frames=15, post_frames=5):
    if all_speed_traces.size == 0:
        print("No speed traces to plot.")
        return

    # Compute mean and standard error
    avg_speed = np.nanmean(all_speed_traces, axis=0)
    sem_speed = np.nanstd(all_speed_traces, axis=0) / np.sqrt(all_speed_traces.shape[0])

    x = np.arange(-pre_frames, post_frames + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(x, avg_speed, color='black', label='Mean Speed')
    plt.fill_between(x, avg_speed - sem_speed, avg_speed + sem_speed, color='gray', alpha=0.3, label='SEM')

    plt.axvline(0, color='blue', linestyle='--', label='Interaction Bout')
    plt.xlabel('Frames relative to interaction')
    plt.ylabel('Running wheel speed')
    plt.title('Running speed during interaction bouts')
    # plt.legend()
    plt.tight_layout()
    plt.show()

```

```python
plot_avg_speed_with_sem(speed_traces)
```

```python
process_and_aggregate_speed_resampled(social_mice, social_2P_sessions)
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
