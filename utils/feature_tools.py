import os 
import sys
import numpy as np 

import scipy.io.wavfile as wav
from python_speech_features import mfcc
import logging
logging.disable(sys.maxsize)

from .feature_helper import * 

def get_wav_vector(wav_path, mfcc_channel=26): 
    fs, audio = wav.read(wav_path)
    total_time = audio.shape[0]/fs
    print(f'Audio {os.path.split(wav_path)[-1]} has {(int(total_time)//60)} min {(total_time)%60:.2f} sec audio length.')

    if len(audio.shape) == 2:
        audio = (audio[:, 0] + audio[:, 1]) / 2

    input_vectors = mfcc(audio, winlen=0.02, winstep=0.01, samplerate=fs, numcep=mfcc_channel) # FPS : 1/0.01 -> 100 
    input_vectors = np.transpose([average(input_vectors[:, i], 5) for i in range(mfcc_channel)]) # FPS : 20 
    
    return input_vectors


def get_bvh_vector(bvh_path, n_joint):
    with open(bvh_path, 'r') as bvh : 
        bvh_content = bvh.readlines()
    hierarchy, bvh_info, bvh_frame = split_bvh_content(bvh_content)
    total_time = int(bvh_info[1].split()[-1]) * float(bvh_info[2].split()[-1])
    print(f'BVH {os.path.split(bvh_path)[-1]} has {int(total_time//60)} min {total_time%60:.2f} sec movement length.')
    
    fps = round(1/float(bvh_info[2].split()[-1]))

    nodes = create_hierarchy_nodes(hierarchy)

    results_bvh_frame = [[float(x) for i, x in enumerate(line.split()) if i > 2] for line in bvh_frame]
    if fps == 100 : # 100 FPS -> 20 FPS (every 5th line)
        results_bvh_frame = results_bvh_frame[::5]
    elif fps == 60 : # 60 FPS -> 20 FPS (every 3rd line)
        results_bvh_frame = results_bvh_frame[::3]
    elif fps == 24 : # 24 FPS -> 20 FPS (del every 6th line)
        del results_bvh_frame[::6]
    
    output_vectors = rot_vec_to_abs_pos_vec(results_bvh_frame, nodes, n_outputs=n_joint)
    return output_vectors

def size_match(vector1, vector2):
    min_len = min(len(vector1), len(vector2))

    vector1 = vector1[:min_len]
    vector2 = vector2[:min_len]

    return vector1, vector2

def save_vectors(base_name, wav_vector, bvh_vector, target_folder): 
    np.save(os.path.join(target_folder, f'{base_name}_X.npy'), wav_vector)
    np.save(os.path.join(target_folder, f'{base_name}_Y.npy'), bvh_vector)
    print(f'{base_name} file saved... wav shape : {wav_vector.shape}, bvh shape : {bvh_vector.shape}')
    print()
