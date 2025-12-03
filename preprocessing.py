import os 
import glob 
import argparse

from utils.feature_tools import * 

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_folder', type=str, help='folder that contains wav and bvh files(or folders)')
    parser.add_argument('-t', '--target_folder', type=str, help='target folder that will hold preprocessed data')

    # audio args 
    parser.add_argument('--mfcc_inputs', type=int, default=26, help='How many features we will store for each MFCC vector')

    # gesture args 
    parser.add_argument('--n_joint', type=int, default=26, help='''Number of joint on motion data. 
                                                                   Output of motion vector size will be determined to n_joint * 3 (x, y, z)''')

    return parser.parse_args()

def main():
    args = parse_args()

    assert os.path.exists(args.data_folder), f'Check your dataset folder {args.data_folder}'
    if not os.path.exists(args.target_folder) : os.makedirs(args.target_folder)

    all_files  = glob.glob(os.path.join(args.data_folder, '**'), recursive=True)

    wavs, bvhs = [], [] 
    for f in all_files : 
        if f.endswith('.wav') : wavs.append(f)
        if f.endswith('.bvh') : bvhs.append(f)

    assert len(wavs) == len(bvhs), f'bvh and wav file number unmatched. num wav : {len(wavs)}, num bvh : {len(bvhs)}'

    name_wav_bvh = [] 
    for wav in wavs: 
        base_file_name = os.path.split(wav)[-1].replace('.wav', '')
        bvh = [b for b in bvhs if base_file_name in b][0]
        name_wav_bvh.append([base_file_name, wav, bvh])

    for idx, (base_name, wav, bvh) in enumerate(name_wav_bvh): 
        wav_vector = get_wav_vector(wav, args.mfcc_inputs)
        bvh_vector = get_bvh_vector(bvh, args.n_joint)
        wav_vector, bvh_vector = size_match(wav_vector, bvh_vector)

        print(f' {idx+1} / {len(name_wav_bvh)} \t ', end = '')
        save_vectors(base_name, wav_vector, bvh_vector, args.target_folder)

if __name__=='__main__':
    main()