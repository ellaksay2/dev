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
plt.figure(figsize=(10,5))
plt.plot(sess.tunnel_data['head_velocity'])
```

```python
import dill
mouse = social_mice[0]
d = social_2P_sessions[mouse][2]
date, scene, session, scan = d['date'], d['scene'], d['session'], d['scan']
pkldir = os.path.join('Z:/giocomo/esay/cd_project/social_interaction/SessPkls', mouse,  date, f'{scene}_{session}.pkl')

with open(pkldir, 'rb') as file:
    sess = dill.load(file)
```

```python
sess.tunnel_data
```

```python

```
