import os

import dill
import numpy as np
import scipy as sp
import pandas as pd
import warnings

import pickle

import TwoPUtils as tpu
import TwoPUtils.utilities as u
from TwoPUtils.utilities import nansmooth

from TwoPUtils import sess
from TwoPUtils import preprocessing as pp
from . import sleap_utils as slp
from . import social_int_sess_deets

social_mice = social_int_sess_deets.social_mice

def align_VR_to_2P(self, overwrite=True, run_ttl_check = False):
        
    if self.vr_data is None or overwrite:
        # load sqlite file as pandas array
    
        if "wheel" in self.vr_filename:
            df = pp.load_sqlite(self.vr_filename,fix_teleports=False)
        else:
            df = pp.load_sqlite(self.vr_filename,fix_teleports=True)
        
        if not self.VR_only:
            # feed pandas array and scene name to alignment function
            if self.scanner == "NLW":
                self.vr_data = pp.vr_align_to_2P(df, self.scan_info, run_ttl_check = run_ttl_check, n_planes=self.n_planes)
    
                # ES add multi-chan functionality 
                if self.n_channels > 1:
                    self.chan0_vr, self.chan1_vr = vr_align_to_2P(df, self.scan_info, run_ttl_check = run_ttl_check, n_planes=self.n_planes, mux = True)
                    print(self.chan0_vr.shape, self.chan1_vr.shape)
        else:
            self.vr_data = df
    
        self.trial_start_inds = self.vr_data.index[self.vr_data.tstart == 1]
        self.teleport_inds = self.vr_data.index[self.vr_data.teleport == 1]
    else:
        print("VR data already set or overwrite=False")
            
def align_SLEAP_to_2P(sess, sleap_dir = "C:/Users/esay/data/social_interaction/SLEAP_raw/videos", overwrite=True, run_ttl_check = False):
    sleap_file = os.path.join(sleap_dir, sess.mouse, sess.date, (sess.scene +'.h5') )
    
    if sess.tunnel_df is None or overwrite:
        if 'Env1' in sess.vr_filename:
            return
        elif "empty" in sess.vr_filename:
            return
        else:
            df = slp.add_tunnel_sess(sleap_file, sess)
            sess.tunnel_data = tunnel_align_to_2P(df, sess.scan_info, run_ttl_check = run_ttl_check, n_planes=sess.n_planes)
            
def vr_align_to_2P(vr_dataframe, scan_info, run_ttl_check=False, n_planes = 1, mux=False):
    """
    Align VR data to 2P scanning data, for NLW rig
    :param vr_dataframe: VR SQLite data as a pandas dataframe
    :param scan_info: scanning metadata from Scanbox .mat file
    :param run_ttl_check: whether to check for aberrant TTLs from poor grounding
    :param n_planes: number of imaged planes
    :param multi_chan: multi channel functionality #ES
    :return: dataframe with one row per imaging frame, containing aligned/interpolated VR data
    """

    # TTLs coming from Unity to the scanbox computer are stored as frame and line
    # indices in the scanbox .mat file, loaded here as scan_info['frames'] and scan_info['lines'].
    # In MATLAB, this would be info.frames and info.lines after loading the .mat file.
    # For instance, if the second Unity TTL arrived at imaging frame 3, line 112 (e.g. out of 512),
    # then scan_info['frame_rate'][1]=3 and scan_info['lines'][1]=112. 

    # Use the frame rate and line rate to estimate timestamps at which each of these TTLs arrived,
    # relative to the start of the imaging session.
    fr = scan_info['frame_rate'] # frame rate
    
    lr = fr * scan_info['config']['lines']/scan_info['fov_repeats']  # line rate

    if 'frame' in scan_info.keys() and 'line' in scan_info.keys():
        frames = scan_info['frame'].astype(int)
        frame_diff = np.ediff1d(frames, to_begin=0)
        try:
            mods = np.argwhere(frame_diff < -100)[0]
            for i, mod in enumerate(mods.tolist()):
                frames[mod:] += (i + 1) * 65535
        except:
            pass
        frames = frames * scan_info['fov_repeats']

        # frames = np.array([f * scan_info['fov_repeats'] for f in scan_info['frame']])
        if scan_info['fold_lines']>0:
            lines = np.array([l % scan_info['fold_lines'] for l in scan_info['line']])
        else:
            lines = np.array(scan_info['line'])
    else:
        # frames = np.array([f * scan_info['fov_repeats'] for f in scan_info['frames']])
        frames = scan_info['frames'].astype(np.int)
        frame_diff = np.ediff1d(frames, to_begin=0)
        try:
            mods = np.argwhere(frame_diff < -100)[0]
            for i, mod in enumerate(mods.tolist()):
                frames[mod:] += (i + 1) * 65535
        except:
            pass
        frames = frames * scan_info['fov_repeats']
        # lines = np.array([l % scan_info['fold_lines'] for l in scan_info['lines']])
        if scan_info['fold_lines']>0:
            lines = np.array([l % scan_info['fold_lines'] for l in scan_info['lines']])
        else:
            lines = np.array(scan_info['lines'])
    

    print(f"frame rate {fr}")
    # Estimate the TTL timestamps
    ttl_times = frames / fr + lines / lr
    # print(ttl_times[-100:])

    if run_ttl_check:
        mask = _ttl_check(ttl_times)
        print('bad ttls', mask.sum())
        ttl_times = ttl_times[mask]
        frames = frames[mask]
        # lines = lines[mask]


    numVRFrames = frames.shape[0]
    # print('numVRFrames', numVRFrames)

    # create empty pandas dataframe to store calcium aligned data
    ca_df = pd.DataFrame(columns=vr_dataframe.columns, index=np.arange(int(scan_info['max_idx']/n_planes)))

    ## create an evenly spaced timeseries for the aligned frames
    # ca_time = np.arange(0, 1 / fr * scan_info['max_idx'], 1 / fr)
    ca_time = np.arange(0,1/fr * scan_info['max_idx'], n_planes/fr)
    ca_time[ca_time>ttl_times[-1]]=ttl_times[-1]

    print(f"{ttl_times.shape} ttl times,{ca_time.shape} ca2+ frame times")
    print(f"last time: VR {ttl_times[-1]}, ca2+ {ca_time[-1]}")
    if (ca_time.shape[0] - ca_df.shape[0]) == 1:  # occasionally a 1 frame correction due to
        # scan stopping mid frame
        warnings.warn('one frame correction')
        ca_time = ca_time[:-1]

    ca_df.loc[:, 'time'] = ca_time
    mask = ca_time >= ttl_times[0]  # mask for when ttls have started on imaging clock
    # (i.e. imaging started and stabilized, ~10s)

    # take VR frames for which there are valid TTLs
    vr_dataframe = vr_dataframe.iloc[-numVRFrames:]

    #find columns that exist in sqlite file from iterable
    column_filter = lambda columns: [col for col in vr_dataframe.columns if col in columns]

    # Below, use interpolation to "downsample" the behavior data to match the times of the imaging data

    # linear interpolation of position and catmull rom spline "time" parameter
    lin_interp_cols = column_filter(('pos','posx','posy','t')) #'posx','posy',

    f_mean = sp.interpolate.interp1d(ttl_times, vr_dataframe[lin_interp_cols]._values, axis=0, kind='slinear')
    ca_df.loc[mask, lin_interp_cols] = f_mean(ca_time[mask])
    ca_df.loc[~mask, 'pos'] = -500.

    # nearest frame interpolation
    near_interp_cols = column_filter(('morph', 'towerJitter', 'wallJitter',
                                      'bckgndJitter','trialnum','cmd','scanning','dreamland', 'LR'))

    f_nearest = sp.interpolate.interp1d(ttl_times, vr_dataframe[near_interp_cols]._values, axis=0, kind='nearest')
    ca_df.loc[mask, near_interp_cols] = f_nearest(ca_time[mask])
    ca_df.fillna(method='ffill', inplace=True)
    ca_df.loc[~mask, near_interp_cols] = -1.

    # integrate, interpolate and then take difference, to make sure data is not lost

    # Note that for licks, taking a cumulative count per imaging frame corresponds to the 
    # number of VR frames where the capacative sensor remained at 1, which should not be 
    # interpreted literally as a number of complete licks per imaging frame, but as an
    # approximation of the rate.
    cumsum_interp_cols = column_filter(('dz', 'lick', 'reward', 'tstart', 'teleport', 'rzone'))
    f_cumsum = sp.interpolate.interp1d(ttl_times, np.cumsum(vr_dataframe[cumsum_interp_cols]._values, axis=0), axis=0,
                                       kind='slinear')
    ca_cumsum = np.round(np.insert(f_cumsum(ca_time[mask]), 0, [0]*len(cumsum_interp_cols), axis=0))
    if ca_cumsum[-1, -2] < ca_cumsum[-1, -3]:
        ca_cumsum[-1, -2] += 1

    ca_df.loc[mask, cumsum_interp_cols] = np.diff(ca_cumsum, axis=0)
    ca_df.loc[~mask, cumsum_interp_cols] = 0.

    # fill na here
    ca_df.loc[np.isnan(ca_df['teleport']._values), 'teleport'] = 0
    ca_df.loc[np.isnan(ca_df['tstart']._values), 'tstart'] = 0
    # if first tstart gets clipped
    if ca_df['teleport'].sum(axis=0) != ca_df['tstart'].sum(axis=0):
        warnings.warn("Number of teleports and trial starts don't match")
        if ca_df['teleport'].sum(axis=0) - ca_df['tstart'].sum(axis=0) == 1:
            warnings.warn(("One more teleport and than trial start, Assuming the first trial start got clipped"))
            ca_df['tstart'].iloc[0]=1

        if ca_df['teleport'].sum(axis=0) - ca_df['tstart'].sum(axis=0) == -1:
            warnings.warn(('One more trial start than teleport, assuming the final teleport got chopped'))
            ca_df['teleport'].iloc[-1]=1
    
    # smooth instantaneous speed
    cum_dz = sp.ndimage.filters.gaussian_filter1d(np.cumsum(ca_df['dz']._values), 5)
    ca_df['dz'] = np.ediff1d(cum_dz, to_end=0)

    # ca_df['speed'].interpolate(method='linear', inplace=True)
    ca_df['speed'] = np.array(np.divide(ca_df['dz'], np.ediff1d(ca_df['time'], to_begin=1. / fr)))
    ca_df['speed'].iloc[0] = 0

    # calculate and smooth lick rate -- note this uses the cumulative lick count per imaging frame,
    # which may produce unnaturally high rates given the small time bin of each imaging frame.
    # For a more conservative estimate of lick rate when we do spatial binning downstream, we will set
    # lick count per imaging frame to 1 if ca_df['lick']>=1. 
    ca_df['lick rate'] = np.array(np.divide(ca_df['lick'], np.ediff1d(ca_df['time'], to_begin=1. / fr)))
    ca_df['lick rate'] = sp.ndimage.filters.gaussian_filter1d(ca_df['lick rate']._values, 5)

    # replace nans with 0s
    ca_df.fillna(value=0, inplace=True)

    '''
    ES add multi_chan functionality, not needed
    '''
    # if ca_df.shape[0] % 2 == 1:
    #         ca_df = ca_df[:-1]
        
    if mux:
        
        chan0_vr  = ca_df.iloc[::2].reset_index(drop = True) # Chan0 even frames
        chan1_vr = ca_df.iloc[1::2].reset_index(drop=True) # Chan1 odd frames
        return chan0_vr, chan1_vr
    
    
    return ca_df

def tunnel_align_to_2P(tunnel_df, scan_info, run_ttl_check=False, n_planes = 1, mux=False):
    """
    Align VR data to 2P scanning data, for NLW rig
    :param tunnel_df: tunnel data as a pandas dataframe
    :param scan_info: scanning metadata from Scanbox .mat file
    :param run_ttl_check: whether to check for aberrant TTLs from poor grounding
    :param n_planes: number of imaged planes
    :param multi_chan: multi channel functionality #ES
    :return: dataframe with one row per imaging frame, containing aligned/interpolated VR data
    """

    # TTLs coming from Unity to the scanbox computer are stored as frame and line
    # indices in the scanbox .mat file, loaded here as scan_info['frames'] and scan_info['lines'].
    # In MATLAB, this would be info.frames and info.lines after loading the .mat file.
    # For instance, if the second Unity TTL arrived at imaging frame 3, line 112 (e.g. out of 512),
    # then scan_info['frame_rate'][1]=3 and scan_info['lines'][1]=112. 

    # Use the frame rate and line rate to estimate timestamps at which each of these TTLs arrived,
    # relative to the start of the imaging session.
    fr = scan_info['frame_rate'] # frame rate
    
    lr = fr * scan_info['config']['lines']/scan_info['fov_repeats']  # line rate

    if 'frame' in scan_info.keys() and 'line' in scan_info.keys():
        frames = scan_info['frame'].astype(int)
        frame_diff = np.ediff1d(frames, to_begin=0)
        try:
            mods = np.argwhere(frame_diff < -100)[0]
            for i, mod in enumerate(mods.tolist()):
                frames[mod:] += (i + 1) * 65535
        except:
            pass
        frames = frames * scan_info['fov_repeats']

        # frames = np.array([f * scan_info['fov_repeats'] for f in scan_info['frame']])
        if scan_info['fold_lines']>0:
            lines = np.array([l % scan_info['fold_lines'] for l in scan_info['line']])
        else:
            lines = np.array(scan_info['line'])
    else:
        # frames = np.array([f * scan_info['fov_repeats'] for f in scan_info['frames']])
        frames = scan_info['frames'].astype(np.int)
        frame_diff = np.ediff1d(frames, to_begin=0)
        try:
            mods = np.argwhere(frame_diff < -100)[0]
            for i, mod in enumerate(mods.tolist()):
                frames[mod:] += (i + 1) * 65535
        except:
            pass
        frames = frames * scan_info['fov_repeats']
        # lines = np.array([l % scan_info['fold_lines'] for l in scan_info['lines']])
        if scan_info['fold_lines']>0:
            lines = np.array([l % scan_info['fold_lines'] for l in scan_info['lines']])
        else:
            lines = np.array(scan_info['lines'])
    

    print(f"frame rate {fr}")
    # Estimate the TTL timestamps
    ttl_times = frames / fr + lines / lr
    # print(ttl_times[-100:])

    if run_ttl_check:
        mask = _ttl_check(ttl_times)
        print('bad ttls', mask.sum())
        ttl_times = ttl_times[mask]
        frames = frames[mask]
        # lines = lines[mask]


    numVRFrames = frames.shape[0]
    # print('numVRFrames', numVRFrames)

    # create empty pandas dataframe to store calcium aligned data
    ca_df = pd.DataFrame(columns=tunnel_df.columns, index=np.arange(int(scan_info['max_idx']/n_planes)))

    ## create an evenly spaced timeseries for the aligned frames
    # ca_time = np.arange(0, 1 / fr * scan_info['max_idx'], 1 / fr)
    ca_time = np.arange(0,1/fr * scan_info['max_idx'], n_planes/fr)
    ca_time[ca_time>ttl_times[-1]]=ttl_times[-1]

    print(f"{ttl_times.shape} ttl times,{ca_time.shape} ca2+ frame times")
    print(f"last time: Tunnel {ttl_times[-1]}, ca2+ {ca_time[-1]}")
    if (ca_time.shape[0] - ca_df.shape[0]) == 1:  # occasionally a 1 frame correction due to
        # scan stopping mid frame
        warnings.warn('one frame correction')
        ca_time = ca_time[:-1]

    ca_df.loc[:, 'time'] = ca_time
    mask = ca_time >= ttl_times[0]  # mask for when ttls have started on imaging clock
    # (i.e. imaging started and stabilized, ~10s)

    # take VR frames for which there are valid TTLs
    tunnel_df = tunnel_df.iloc[-numVRFrames:]

    #find columns that exist in sqlite file from iterable
    column_filter = lambda columns: [col for col in tunnel_df.columns if col in columns]

    # Below, use interpolation to "downsample" the behavior data to match the times of the imaging data

    # linear interpolation of position and catmull rom spline "time" parameter
    lin_interp_cols = column_filter(('head_velocity','rightear_x','rightear_y','tailbase_x','tailbase_y','nose_x','nose_y','head_x','head_y','leftear_x','leftear_y','torso_x','torso_y'))

    f_mean = sp.interpolate.interp1d(ttl_times, tunnel_df[lin_interp_cols]._values, axis=0, kind='slinear')
    ca_df.loc[mask, lin_interp_cols] = f_mean(ca_time[mask])
    ca_df.loc[~mask, 'pos'] = -500.

    # nearest frame interpolation
    near_interp_cols = column_filter(('interaction'))

    f_nearest = sp.interpolate.interp1d(ttl_times, tunnel_df[near_interp_cols]._values, axis=0, kind='nearest')
    ca_df.loc[mask, near_interp_cols] = f_nearest(ca_time[mask])
    ca_df.fillna(method='ffill', inplace=True)
    ca_df.loc[~mask, near_interp_cols] = -1.

    # replace nans with 0s
    ca_df.fillna(value=0, inplace=True)

    '''
    ES add multi_chan functionality, not needed
    '''
    # if ca_df.shape[0] % 2 == 1:
    #         ca_df = ca_df[:-1]
        
    if mux:
        
        chan0_vr  = ca_df.iloc[::2].reset_index(drop = True) # Chan0 even frames
        chan1_vr = ca_df.iloc[1::2].reset_index(drop=True) # Chan1 odd frames
        return chan0_vr, chan1_vr
    
    
    return ca_df