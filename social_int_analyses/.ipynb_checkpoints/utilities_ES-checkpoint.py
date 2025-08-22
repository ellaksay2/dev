import os

import dill
import numpy as np

import pickle

import TwoPUtils as tpu
from TwoPUtils import sess
from . import social_int_sess_deets
from . import alignment

social_mice = social_int_sess_deets.social_mice

def loop_func_over_mice(func, mice):
    return {mouse: func(mouse) for mouse in mice}

def loop_func_over_days(func, days, **kwargs):
    return lambda mouse: [func(load_single_day(mouse, day), **kwargs) for day in days]


def common_rois(roi_matches, inds):
    ref = roi_matches[inds[0]]
    ref_common_rois = []

    for i, targ_ind in enumerate(inds[1:]):

        #         targ = roi_matches[targ_ind][inds[0]]
        if i == 0:

            ref_common_rois = set(ref[targ_ind]['ref_inds'])
        else:
            ref_common_rois = ref_common_rois & set(ref[targ_ind]['ref_inds'])

        # find cells that are in reference match list each time
    ref_common_rois = list(ref_common_rois)

    # find matching indices
    common_roi_mapping = np.zeros([len(inds), len(ref_common_rois)]) * np.nan
    common_roi_mapping[0, :] = ref_common_rois
    for i, roi in enumerate(ref_common_rois):
        for j, targ_ind in enumerate(inds[1:]):
            #             print(j)
            ind = np.argwhere(ref[targ_ind]['ref_inds'] == roi)[0][0]
            #             print(j,roi,ind)
            common_roi_mapping[j + 1, i] = ref[targ_ind]['targ_inds'][ind]

    return common_roi_mapping.astype(int)



def get_ind_of_exp_day(sess_list, exp_day):

    index = [] # default if no matching exp_day is found

    for ind, day_dict in enumerate(sess_list):
        if type(day_dict) is tuple:
            for i, entry in enumerate(day_dict):
                # print(i)
                if entry["exp_day"] == exp_day:
                    index = ind
        else:
            if day_dict["exp_day"] == exp_day:
                index = ind

    return index

def load_vr_day(mouse,day, verbose = True, trial_mat_keys = ('licks','speed'), timeseries_keys = ('licks', 'speed')):
    if mouse in social_mice:
        pkldir = os.path.join('C:/Users/esay/data/social_interaction/VRPkls', mouse)
        deets = social_int_sess_deets.social_VR_sessions[mouse][day]
        
    else:
        raise Exception("invalid mouse name")

    if verbose:
        print(deets)
    if isinstance(deets, tuple):

        sess_list = []
        for _deets in deets:
            _sess = tpu.sess.Session.from_file(
                os.path.join(pkldir, _deets['date'], "%s_%d.pkl" % (_deets['scene'], _deets['session'])),
                verbose=False)

            sess_list.append(_sess)
    else:
        pkl_path = os.path.join(pkldir, deets['date'], "%s_%d.pkl" % (deets['scene'], deets['session']))
        sess = pickle.load(open(pkl_path, "rb"))

    return sess

def load_single_day(mouse, day, pkl_basedir = "C://Users/esay/data/Stx3/YMazeSessPkls",verbose = True):
    pkldir = os.path.join(pkl_basedir, mouse)
    
    if mouse in sparse_mice:
        deets = social_int_sess_deets.social_sessions[mouse][day]
    else:
        raise Exception("invalid mouse name")

    if verbose:
        print(deets)
    
    pkl_path = os.path.join(pkldir, deets['date'], "%s_%d.pkl" % (deets['scene'], deets['session']))
    sess = pickle.load(open(pkl_path, "rb"))
    sess.add_timeseries(licks=sess.vr_data['lick']._values)
    sess.add_pos_binned_trial_matrix('licks')
    return sess




def is_putative_interneuron(sess, ts_key='dff', method='speed',
                            prct=10, r_thresh=0.3):
    """
    Find putative interneurons based on spatially-binned
    trial_mat values, ratio of 99th prctile to mean, per cell.
    Taking anything <10th prctile within animal as an "int".

    :param sess: session class
    :param ts_key: key of timeseries to use
    :param method: method of identifying ints: 'speed' or 'trial_mat_ratio'
    :param prct: (for method 'trial_mat_ratio') percentile of ROIs within animal to cut off
    :param r_thresh: (for method 'speed') threshold for speed correlation r
    :return: is_int (list) with a Boolean for each ROI, where
        putative interneuron is True.
    """

    use_trial_mat = np.copy(sess.trial_matrices[ts_key][0])
    if method == 'trial_mat_ratio':
        trial_mat_prop = []
        trial_mat_ratio = []
        for c in range(use_trial_mat.shape[2]):
            _trial_mat = use_trial_mat[:, :, c]
            nanmask = np.isnan(_trial_mat)
            trial_mat_prop.append(np.percentile(_trial_mat[~nanmask], 75))

            trial_mat_max = np.percentile(_trial_mat[~nanmask], 99)

            ratio = trial_mat_max / np.mean(_trial_mat[~nanmask])
            trial_mat_ratio.append(ratio)

        is_int = trial_mat_ratio < np.percentile(trial_mat_ratio, prct)

    elif method == 'speed':
        speed_corr = np.zeros((use_trial_mat.shape[1],))
        speed = np.copy(sess.vr_data['speed']._values)
        nanmask = ~np.isnan(sess.timeseries[ts_key][0, :])

        for c in range(use_trial_mat.shape[1]):
            speed_corr[c] = np.corrcoef(
                sess.timeseries[ts_key][c, nanmask], speed[nanmask])[0, 1]

        is_int = speed_corr < r_thresh

    return is_int

def update_sess_dict(mouse, f, KO = True):
    scan_str = "%s_%03d_%03d" % (f['scene'],f['session'] ,f['scan'])
    
    source_folder = "C:/Users/esay/data/social_interaction/VRData"
    source_stem = os.path.join(source_folder, mouse, f['date'], f['scene'], scan_str)

    f['mouse']=mouse
    f.update({'vr_filename': os.path.join("C:/Users/esay/data/social_interaction/VRData",f['mouse'],f['date'],"%s_%d.sqlite" %(f['scene'],f['session'])),
              'scan_number': f['scan'],
              'prompt_for_keys': False,
              'VR_only': True,
              'scanner': "NLW",
             })
    if 'wheel' in f['scene']:
        f.update(
            {'fixed_wheel': True
            })
    if 'social' in f['scene']:
        # print(f['scene'])
        f.update({
            'tunnel_data': True
        })
    if 'emptytunnel' in f['scene']:
        # print(f['scene'])
        f.update({
            'tunnel_data': False
        })
    return f

def run_and_save(fdict,comb_frames=0, social = True):
    sess = tpu.sess.Session(**fdict)
    
    alignment.align_VR_to_2P(sess)
    # sess.get_trial_info()
    
    sess.add_timeseries(speed_raw=sess.vr_data['dz']._values)
    sess.add_pos_binned_trial_matrix('speed_raw', pos_key = 'pos')
    
    sess.add_timeseries(speed=sess.vr_data['dz']._values)
    sess.add_pos_binned_trial_matrix('speed', pos_key = 'pos', speed_thr=.01, speed = sess.timeseries['speed_raw'].ravel())

    if social == False: 
        err_lick, err = correct_lick_sensor_error(sess.vr_data['lick'],sess.trial_start_inds, sess.teleport_inds, correction_thr=0.5) #correct for broken sensor
        sess.add_timeseries(licks_rate = err_lick)
        sess.add_pos_binned_trial_matrix('licks_rate', pos_key = 'pos')# ,speed_thr=.01, speed = sess.timeseries['speed_raw'].ravel())
        sess.add_timeseries(licks=err_lick)
        sess.timeseries['licks'][sess.timeseries['licks']>1]=1
        sess.add_pos_binned_trial_matrix('licks', pos_key = 'pos')
        
        licks = np.copy(sess.timeseries['licks'])
        licks[licks>1]=1
        time = sess.vr_data['time']
    
        reward_inds = np.argwhere(sess.vr_data['reward']._values==1).ravel()
        for r in reward_inds:
            licks[0,((time-time[r])>=0)&((time-time[r])<2)] = 0
    
        sess.add_timeseries(nonconsum_licks = licks)
        sess.add_pos_binned_trial_matrix('nonconsum_licks', pos_key = 'pos')
        
        sess.add_timeseries(licks_sum=licks)
        sess.add_pos_binned_trial_matrix('licks_sum',use_sum=True, pos_key = 'pos')
        
        for key in ('licks_rate', 'licks', 'nonconsum_licks', 'licks_sum'):
            print(np.isnan(sess.trial_matrices[key][0]).sum())
        tpu.sess.save_session(sess,'C:/Users/esay/data/social_interaction/VRPkls')
    
    tpu.sess.save_session(sess,'C:/Users/esay/data/social_interaction/VRPkls')
    return comb_frames+sess.vr_data.shape[0]


def write_sess_pickle(sess, sess_dir, pkl_name, overwrite=False):
    """
    Write a sess class to a .pickle file

    :param sess: session class
    :param sess_dir: directory to write the pickle file
    :param pkl_name: pickle file name
    :param overwrite: whether to overwrite an existing pickle file with this name, if it exists
    :return:
    """

    print("writing", pkl_name)

    if overwrite:
        save_sess = open(os.path.join(sess_dir, pkl_name), "wb")
        pickle.dump(sess, save_sess)
    else:
        if os.path.exists(os.path.join(sess_dir, pkl_name)):
            raise NotImplementedError(
                ".pickle already exists, aborting save. Set overwrite=True to overwrite."
            )
        else:
            save_sess = open(os.path.join(sess_dir, pkl_name), "wb")
            pickle.dump(sess, save_sess)
            # Close the pickle
    save_sess.close()

    return
    
def correct_lick_sensor_error(licks_, trial_starts, trial_ends, correction_thr=0.5):
    """
    Find samples where lick detector was putatively stuck at 1, and set to NaN

    :param licks_:
    :type licks_:
    :param trial_starts:
    :type trial_starts:
    :param trial_ends:
    :type trial_ends:
    :param correction_thr:
    :type correction_thr:
    :return:
    :rtype:
    """
    licks = np.copy(licks_)
    error_count = 0
    for t_start, t_end in zip(trial_starts, trial_ends):
        # if >50% of samples have a cumulative lick count of >2
        # print(licks[t_start:t_end])
        if sum(licks[t_start:t_end] > 0)/len(licks[t_start:t_end]) > correction_thr:
            licks[t_start:t_end] = 0
            # print(f'setting trial {np.where(trial_starts==t_start)[0]} to NaN')
            error_count += 1

    return licks, error_count