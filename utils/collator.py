import os 
import torch 
import numpy as np

from utils.utils import get_numpy

class Collator():
    def __init__(self, silence_npy_path, bvh_npy_path):
        self.silence_npy_path = silence_npy_path
        self.bvh_npy_path = bvh_npy_path

    def collate_fn(self, batch):
        wavs, xyzs = zip(*batch)
        wavs, xyzs = list(wavs), list(xyzs)

        if len(wavs) != 1 :
            wav_pad_base = get_numpy(self.silence_npy_path)
            xyz_pad_base = get_numpy(self.bvh_npy_path)
            
            lengths = []
            for wav in wavs:
                lengths.append(len(wav))
            max_length = max(lengths)

            for idx in range(len(wavs)) : 
                num_path = max_length - len(wavs[idx])
                
                wav_pad = np.repeat(wav_pad_base, num_path, axis=0)
                wavs[idx] = np.append(wavs[idx], wav_pad, axis=0)

                xyz_pad = np.repeat(xyz_pad_base, num_path, axis=0)
                xyzs[idx] = np.append(xyzs[idx], xyz_pad, axis=0)

        return (torch.tensor(wavs), torch.tensor(xyzs))
