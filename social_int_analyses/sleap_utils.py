import os
import math   
import h5py
import os
import numpy as np # type: ignore
import pandas as pd # type: ignore
from scipy.interpolate import interp1d

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

import TwoPUtils as tpu

# import videos
def list_mp4_files(directory):
    # List to store mp4 file names
    mp4_files = []

    # Iterate through files in the given directory
    for filename in os.listdir(directory):
        # Check if file is an mp4
        if filename.endswith('.mp4'):
            mp4_files.append(filename)
    
    return mp4_files

    
# Change the brightness and contrast here
brightness_value = 0.1 
contrast_value = 1.2   

# make bash script to edit brightness and contrast using ffmpeg
def edit_video(dir, output, mp4_files, brightness=0.0, contrast=1.0): #defaults brightness 0.0 default contrast 1.0
    # Create a bash script file
    script_path = os.path.join(dir, 'edit_videos.sh')
    
    with open(script_path, 'w') as script_file:
        # Write the bash script header
        script_file.write("#!/bin/bash\n\n")

        # Loop through each mp4 file and generate ffmpeg commands
        for video in mp4_files:
            filename = os.path.splitext(video)[0]
            input_path = os.path.join(dir, video)
            output_path = os.path.join(output, f"{filename}_edited.mp4")
            
            # Generate ffmpeg command to adjust brightness and contrast
            # ffmpeg_command = f"ffmpeg -i \"{input_path}\" -vf eq=brightness={brightness}:contrast={contrast} \"{output_path}\"\n"

            # Generate ffmpeg command to convert file type
            ffmpeg_command = f"ffmpeg -y -i \"{input_path}\" -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 23 \"{output_path}\"\n"
            
            # Write the command to the bash script
            script_file.write(ffmpeg_command)

    # Make the bash script executable
    os.chmod(script_path, 0o755)
    print(f"Bash script created: {script_path}")

def create_inference_bash(directory, mp4_files, model_path, batch_size=4):
    # Calculate the number of scripts needed
    num_scripts = math.ceil(len(mp4_files) / batch_size)

    # Write the bash scripts
    for script_index in range(num_scripts):
        # Create a bash script file
        script_filename = f"{str(script_index+1).zfill(2)}_inference.bash" #name scripts
        script_path = os.path.join(directory, script_filename)

        with open(script_path, 'w') as script_file:
            # Write the bash script header
            script_file.write("#!/bin/bash\n\n")

            # Write sleap-track commands for a batch of videos
            start_index = script_index * batch_size
            end_index = min(start_index + batch_size, len(mp4_files))

            for i in range(start_index, end_index):
                video_path = os.path.join(directory, mp4_files[i])
                sleap_command = (
                    f"sleap-track \"{video_path}\" "
                    f"-m \"{model_path}\"\n"

                    # Convert .slp files to .h5 files for analysis
                    f"sleap-convert \"{video_path}.predictions.slp\" "
                    f"--format analysis \n"
                )
                # Write the command to the bash script
                script_file.write(sleap_command)

        # Make the bash script executable
        os.chmod(script_path, 0o755)

    print(f"Created {num_scripts} bash scripts in {directory}")

def create_inference_batch(directory, mp4_files, model_path, batch_size=4):
    # Calculate the number of scripts needed
    num_scripts = math.ceil(len(mp4_files) / batch_size)

    # Write the batch scripts
    for script_index in range(num_scripts):
        # Create a batch script file 
        script_filename = f"{str(script_index+1).zfill(2)}_inference.bat" #name scripts

        script_path = os.path.join(directory, script_filename)

        with open(script_path, 'w') as script_file:

            # Write the batch script header (WINDOWS)
            script_file.write("@echo off\n\n")


            # Write sleap-track commands for a batch of videos
            start_index = script_index * batch_size
            end_index = min(start_index + batch_size, len(mp4_files))

            for i in range(start_index, end_index):
                video_path = os.path.join(directory, mp4_files[i])
                sleap_command = (
                    f"sleap-track \"{video_path}\" "
                    f"-m \"{model_path}\" "
                    f"-o \"{video_path}.predictions.slp\n"

                    # Convert .slp files to .h5 files for analysis
                    f"sleap-convert \"{video_path}.predictions.slp\" "
                    f"--format analysis \n"
                )
                # Write the command to the bash script
                script_file.write(sleap_command)

        # Make the bash script executable
        os.chmod(script_path, 0o755)

    print(f"Created {num_scripts} bash scripts in {directory}")

def import_single_slp(filename):
    data = {}
    data['locations'] = []
    
    with h5py.File(filename, "r") as f:
        dset_names = list(f.keys())
        locations = f["tracks"][:].T
        node_names = [n.decode() for n in f["node_names"][:]]

    data['locations'] = locations 
    
    print("===filename===")
    print(filename)
    print()

    print("===HDF5 datasets===")
    print(dset_names)
    print()

    # print("===locations data shape===")
    # print(locations.shape)
    # print()

    # print("===nodes===")
    # for i, name in enumerate(node_names):
    #     print(f"{i}: {name}")
    # print()

    
    return data

def import_h5_dir(directory):
    # Get a list of all .h5 files in the directory
    h5_files = [f for f in os.listdir(directory) if f.endswith('.h5')]
    print(h5_files)
    # Initialize an empty list to store data for the DataFrame
    data = []

    for h5_file in h5_files:
        print(h5_file)
        filename = os.path.join(directory, h5_file)
        # print(filename)
        # Open and process the .h5 file
        with h5py.File(filename, "r") as f:
            # dset_names = list(f.keys())
            locations = f["tracks"][:].T
            # print(locations)
            node_names = [n.decode() for n in f["node_names"][:]]
            
            # Append the data to the list
            condition = extract_condition(h5_file)
            # print(condition)
            # filename = extract_filename(h5_file)
            data.append({
                'filename': filename,
                'name': h5_file,
                'location shape': locations.shape,
                'locations': locations,
                'condition': condition
            })
    return data


def extract_condition(filename):
    start = filename.find("_") + 1
    end = filename.find(".", start)
    if start > 1 and end > -1:
        return filename[start:end]
    return None

def extract_filename(filename):
    start = 0
    end = filename.find(".h5", start)
    return filename[start:end]
  


# Fill in missing values (NaN vavlues due to tracking errors)
def fill_missing(Y, kind="linear"):
    """Fills missing values independently along each dimension after the first."""

    # Store initial shape.
    initial_shape = Y.shape

    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))

    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]

        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)
        
        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y

NOSE_INDEX = 2
LEFTEAR_INDEX = 4
RIGHTEAR_INDEX = 0
TORSO_INDEX = 5
TAILBASE = 1
HEAD_INDEX = 3

def store_nodes(row):
    nodes = {
        'rightear':RIGHTEAR_INDEX,
        'tailbase':TAILBASE,
        'nose':NOSE_INDEX,
        'head':HEAD_INDEX,
        'leftear': LEFTEAR_INDEX,
        'torso': TORSO_INDEX
    }
    location_dict = {}
    locations = row['locations']

    for node, index in nodes.items():

        x_coords = locations[:,index,0,0]
        y_coords = locations[:,index,1,0]
        location_dict[f'{node}_x'] = x_coords
        location_dict[f'{node}_y'] = y_coords
        # node_loc = locations[:,index,:,:].reshape(-1)
        # location_dict[node] = node_loc

    return pd.Series(location_dict)

from scipy.signal import savgol_filter

def smooth_diff(node_loc, win=25, poly=3):
    """
    node_loc is a [frames, 2] array
    
    win defines the window to smooth over
    
    poly defines the order of the polynomial
    to fit with
    
    """
    node_loc_vel = np.zeros_like(node_loc)
    
    for c in range(node_loc.shape[-1]):
        node_loc_vel[:, c] = savgol_filter(node_loc[:, c], win, poly, deriv=1)
    
    node_vel = np.linalg.norm(node_loc_vel,axis=1)

    return node_vel

def pad_vr_data(locations, sess):
    sess_frames = sess.vr_data.shape[0]
    loc_frames = locations.shape[0]

    if loc_frames >= sess_frames:
        print("Warning: VR does not have more frames than tunnel data")
        return

    padding = sess_frames - loc_frames

    nan_frames = np.zeros((padding, locations.shape[1], locations.shape[2], locations.shape[3]))
    locations = np.concatenate((nan_frames, locations),axis=0)
    print("Padding locations. New shape:", locations.shape)

    return locations

def add_tunnel_sess(h5_path, sess):
    
    # import pre-process sleap h5 file
    df = import_single_slp(h5_path)
    
    #interpolate over missing values
    df['locations'] = fill_missing(df['locations'])

    # pad locations data to equal len of vr data 
    # df['locations'] = pad_vr_data(df['locations'], sess)

    # caluclate head velocity 
    head_loc = df['locations'][:, HEAD_INDEX, :, :]
    head_vel = smooth_diff(head_loc[:, :, 0])
    df['head_velocity'] = head_vel

    # store individual node x and y values 
    nodes = store_nodes(df)
    nodes_df = pd.DataFrame(nodes.tolist()).T
    nodes_df.columns = nodes.index

    # TODO: quantify amount of time spent on the sides vs the middle

    # Quantify time spent in 'interaction zone'
    filtered_frames = []
    
    head_x = np.array(nodes_df['head_x']).astype(float)
    head_y = np.array(nodes_df['head_y']).astype(float)
    
    # interaction zone x = [300,400] y=[200,300]
    # int_zone = (nose_x >= 300) & (nose_x <= 400) & (nose_y >= 200) & (nose_y <= 300)
    int_zone = (head_x >= 300) & (head_x <= 400) & (head_y >= 200) & (head_y <= 250)
    
    frame_indices = np.where(int_zone)[0]  
    filtered_frames.append(frame_indices)
    
    df['interaction'] = int_zone

    # store key points in og dataframe 
    keypoints = {
        'rightear':RIGHTEAR_INDEX,
        'tailbase':TAILBASE,
        'nose':NOSE_INDEX,
        'head':HEAD_INDEX,
        'leftear': LEFTEAR_INDEX,
        'torso': TORSO_INDEX
    }
    
    location_dict = {}
    locations = df['locations']
    
    for node, index in keypoints.items():
    
        x_coords = locations[:,index,0,0]
        y_coords = locations[:,index,1,0]
        df[f'{node}_x'] = x_coords
        df[f'{node}_y'] = y_coords

    sess.tunnel_df = df
    print(sess.tunnel_df.keys())
    tpu.sess.save_session(sess,'C:/Users/esay/data/social_interaction/VRPkls')

    tunnel_data = {key: value for key, value in df.items() if key !='locations'}
    df = pd.DataFrame(tunnel_data)
    return df 


# plotting utisl

def plot_trace(df):
    # df = sess.tunnel_data
    plt.figure(figsize=(7,7))
    plt.plot(df['nose_x'],1*df['nose_y'], 'b',label='Nose')
    # plt.plot(df['leftear_x'][i],-1*df['leftear_y'][i], 'b',label='Left ear')
    # plt.plot(df['rightear_x'][i],-1*df['rightear_y'][i], 'r',label='Right ear')
    plt.plot(df['head_x'],1*df['head_y'], 'g',label='Head')