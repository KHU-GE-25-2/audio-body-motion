import os 
import argparse
import numpy as np

from utils.feature_tools import get_wav_vector 
from utils.feature_helper import create_hierarchy_nodes

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--wav_path', type=str, help='path to silence.wav file')
    parser.add_argument('--bvh_path', type=str, help='random bvh file used in preprocessing.py')
    parser.add_argument('--target_folder', type=str, default='reference_data', help='folder to contain results npy files')

    parser.add_argument('--mfcc_inputs', type=int, default=26, help='How many features we will store for each MFCC vector')

    return parser.parse_args()

def main(): 
    args = parse_args()

    assert os.path.exists(args.wav_path), f'Check your silence.wav file path.. {args.wav_path}'
    assert os.path.exists(args.bvh_path), f'Check your bvh file path.. {args.bvh_path}'
    if not os.path.exists(args.target_folder) : os.makedirs(args.target_folder)

    ## wav 
    wav_npy = get_wav_vector(args.wav_path, args.mfcc_inputs)
    one_wav_npy = np.expand_dims(wav_npy[0], 0)
    np.save(os.path.join(args.target_folder, 'silence.npy'), one_wav_npy)
    print(f'save reference wav file. shape : {one_wav_npy.shape}')

    ## bvh 
    with open(args.bvh_path, 'r') as bvh : 
        data = bvh.readlines() 

    for idx, d in enumerate(data): 
        if d.startswith('}'):
            break

    data = data[:idx+1] # get base pose part 
    nodes = create_hierarchy_nodes(data)

    root_offset = np.array([0, 60, 0])
    offset_list = [
        root_offset # ROOT offest and to make it [0, 60, 0]
    ] 

    for idx in range(1, len(nodes)) : 
        parent_idx = nodes[idx]['parent']
        offset_list.append(
            offset_list[parent_idx] + nodes[idx]['offset'] 
        )

    one_bvh_npy = np.array(offset_list).flatten()
    one_bvh_npy = np.expand_dims(one_bvh_npy, 0)
    np.save(os.path.join(args.target_folder, 'hierarchy.npy'), one_bvh_npy)
    print(f'save reference bvh file. shape : {one_bvh_npy.shape}')

if __name__=='__main__':
    main()