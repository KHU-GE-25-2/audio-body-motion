import os
import torch
import numpy as np

from torch.utils.data.dataset import Dataset

from utils.utils import get_numpy

class AudioGestureDataset(Dataset):
    def __init__(self, data_folder, context, silence_npy_path):
        self.data_folder = data_folder
        npys = os.listdir(self.data_folder)

        self.X_npys = []
        self.Y_npys = []
        for npy in npys :
            if npy[-5] == 'X' :
                self.X_npys.append(npy)
                assert npy[:-5]+'Y.npy' in npys, "Something Wrong on processed npy results"
                self.Y_npys.append(npy[:-5]+'Y.npy')

        assert len(self.X_npys) == len(self.Y_npys), "Num of input X mis-matched to num of output Y"

        self.wav_pad_base = get_numpy(silence_npy_path)
        self.wav_pad_base = np.repeat(self.wav_pad_base, repeats=context, axis=0)
    
    def __getitem__(self, index) :
        wav = np.load(os.path.join(self.data_folder, self.X_npys[index]))
        xyz = np.load(os.path.join(self.data_folder, self.Y_npys[index]))

        wav = np.append(self.wav_pad_base, wav, axis=0)
        wav = np.append(wav, self.wav_pad_base, axis=0)

        return wav, xyz

    def __len__(self) :
        return len(self.X_npys)
    
class AudioGestureDatasetRevised(Dataset):
    def __init__(self, data_folder, context, silence_npy_path, stats_path='preprocessed_ref'):
        self.data_folder = data_folder
        self.context = context
        
        # 1. LOAD STATISTICS (Critical for Revised Model)
        # We try to load the mean/std we calculated earlier.
        # If not found, we warn the user (model will struggle without this).
        mean_file = os.path.join(stats_path, 'motion_mean.npy')
        std_file = os.path.join(stats_path, 'motion_std.npy')

        if not os.path.exists(mean_file) or not os.path.exists(std_file):
            raise FileNotFoundError(f"[Dataset Error] Stats files not found in '{stats_path}'. Run calculate_stats.py first!")

        self.mean = np.load(mean_file)
        self.std = np.load(std_file)
        print(f"[Dataset] Loaded stats from: {stats_path}")
        print(f"   Mean[0]: {self.mean[0]}")
        print(f"   Std[0]:  {self.std[0]}")

        # 2. FILE PAIRING LOGIC (Your logic)
        npys = sorted(os.listdir(self.data_folder)) # Sorted ensures deterministic order
        self.X_npys = []
        self.Y_npys = []
        
        for npy in npys:
            # Check for 'X' at the specific index, assume format "...X.npy"
            if npy.endswith('X.npy'):
                target_y = npy[:-5] + 'Y.npy'
                if target_y in npys:
                    self.X_npys.append(npy)
                    self.Y_npys.append(target_y)
                else:
                    print(f"Warning: Found {npy} but missing {target_y}")

        assert len(self.X_npys) > 0, "No paired X/Y numpy files found!"
        print(f"[Dataset] Loaded {len(self.X_npys)} pairs.")

        # 3. SETUP PADDING
        # Load silence frame
        raw_silence = get_numpy(silence_npy_path)
        # Ensure it's 2D (1, 26)
        if len(raw_silence.shape) == 1:
            raw_silence = raw_silence[np.newaxis, :]
            
        # Repeat for context length
        self.wav_pad_base = np.repeat(raw_silence, repeats=context, axis=0)
    
    def __getitem__(self, index):
        # Load Raw Data
        wav = np.load(os.path.join(self.data_folder, self.X_npys[index]))
        xyz = np.load(os.path.join(self.data_folder, self.Y_npys[index]))

        # --- SAFETY 1: SANITIZE ---
        # Replace NaNs or Infs with safe numbers to prevent crashes
        wav = np.nan_to_num(wav, nan=0.0)
        xyz = np.nan_to_num(xyz, nan=0.0)

        # --- SAFETY 2: LENGTH CHECK ---
        # Ensure audio and motion are exactly the same length before padding
        min_len = min(wav.shape[0], xyz.shape[0])
        wav = wav[:min_len]
        xyz = xyz[:min_len]

        # --- CRITICAL: NORMALIZE MOTION ---
        # Formula: (Raw - Mean) / Std
        xyz = (xyz - self.mean) / self.std
        
        # Double check for NaNs after normalization (e.g. 0/0 division risk)
        xyz = np.nan_to_num(xyz, nan=0.0)

        # --- PAD AUDIO ---
        # Pad Start and End
        wav = np.concatenate((self.wav_pad_base, wav, self.wav_pad_base), axis=0)

        # Return as FloatTensors
        return torch.FloatTensor(wav), torch.FloatTensor(xyz)

    def __len__(self):
        return len(self.X_npys)