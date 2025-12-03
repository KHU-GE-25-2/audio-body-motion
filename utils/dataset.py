import os
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