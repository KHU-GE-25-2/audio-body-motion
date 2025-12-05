import os
import torch
import glob
import numpy as np
from torch.utils.data.dataset import Dataset
from utils.utils import get_numpy

class AudioGestureDatasetRevised(Dataset):
    def __init__(self, data_dir, context=10, silence_path=None, stats_path=None):
        """
        data_dir: Path to the 'preprocessed_norm' folder.
        stats_path: Path to folder containing stats_X.npz and stats_Y.npz (optional for training, good for debug)
        """
        self.data_dir = data_dir
        self.context = context
        
        # 1. Find all Input (X) files
        # We search for files ending in _X.npy
        self.input_files = sorted(glob.glob(os.path.join(data_dir, "*_X.npy")))
        
        # 2. Match with Output (Y) files
        self.data_pairs = []
        for x_path in self.input_files:
            y_path = x_path.replace("_X.npy", "_Y.npy")
            if os.path.exists(y_path):
                self.data_pairs.append((x_path, y_path))
            else:
                print(f"[Warning] Missing Y file for: {os.path.basename(x_path)}")

        print(f"Dataset Loaded: {len(self.data_pairs)} pairs found in {data_dir}")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        x_path, y_path = self.data_pairs[idx]
        
        # Load the pre-normalized data
        # shape: (Frames, 26) for X, (Frames, 78) for Y
        in_data = np.load(x_path).astype(np.float32)
        out_data = np.load(y_path).astype(np.float32)
        
        # Ensure lengths match exactly (clip the longer one)
        min_len = min(len(in_data), len(out_data))
        in_data = in_data[:min_len]
        out_data = out_data[:min_len]
        
        # Note: If you need to add padding for the context window, do it here.
        # For a standard LSTM, we can usually just feed the raw sequence.
        
        return in_data, out_data

# Maintain the old class just in case imports rely on it, but verify usage in train.py
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