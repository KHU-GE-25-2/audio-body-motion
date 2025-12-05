import os 
import torch 
import torch.nn as nn 

from utils.utils import get_numpy

import numpy as np # Assuming get_numpy is a function that returns a numpy array

# Helper function placeholder
def get_numpy(path):
    # In a real scenario, this would load from the path
    # For this example, let's assume it returns a silent MFCC frame
    return np.zeros((1, 40)) # Example shape (1, mfcc_channel)

# --- A reusable, normalized, and regularized processing block ---
class AudioProcessorBlock(nn.Module):
    """A block with Linear -> LayerNorm -> Activation -> Dropout."""
    def __init__(self, input_dim, output_dim, activation=nn.GELU(), dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

# --- The main, enhanced model ---
class AudioGestureLSTMRevised(nn.Module):
    def __init__(self, input_size, context, hidden_size, output_size, silence_path=None, device='cpu', dropout=0.1):
        super(AudioGestureLSTMRevised, self).__init__()
        
        self.device = device
        self.input_size = input_size    # Should be 26 (MFCC)
        self.output_size = output_size  # Should be 78 (Joints * 3)
        
        # Layer 1: Project Audio Features up to Hidden Size
        self.embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )
        
        # Layer 2: The Core LSTM
        # batch_first=True means input format is (Batch, Time, Features)
        self.lstm = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            num_layers=2, 
            batch_first=True, 
            dropout=dropout
        )
        
        # Layer 3: Project back to Motion Space
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # x shape: (Batch, Frames, 26)
        
        # 1. Embed
        x = self.embedding(x) 
        
        # 2. Recurrent Processing
        # LSTM output shape: (Batch, Frames, Hidden)
        out, _ = self.lstm(x)
        
        # 3. Output
        out = self.regressor(out)
        
        # Result shape: (Batch, Frames, 78)
        return out

class AudioGestureLSTM(nn.Module): 
    def __init__(self, mfcc_channel, context, hidden_size, n_joint, silence_npy_path, device, dropout=0, num_layer=2, bidirectional=True): 
        super().__init__()
        self.mfcc_channel = mfcc_channel
        self.context = context
        self.hidden_size = hidden_size
        self.n_joint = n_joint
        self.dropout = dropout
        self.device = device

        self.num_layer = num_layer
        self.D = 2 if bidirectional else 1 
        
        self.wav_pad_base = torch.tensor(get_numpy(silence_npy_path)).unsqueeze(0).float().to(self.device)  # (1, 1, mfcc_channel)
        
        self.h1 = nn.Conv1d(in_channels=self.mfcc_channel, out_channels=self.mfcc_channel, kernel_size=(2*self.context+1), stride=1)
        self.h2 = nn.Linear(in_features=self.mfcc_channel, out_features=self.mfcc_channel)
        self.h3 = nn.Linear(in_features=self.mfcc_channel, out_features=self.mfcc_channel)
        self.h4 = nn.LSTM(input_size=self.mfcc_channel, hidden_size=self.hidden_size, 
                          num_layers=num_layer, bidirectional=bidirectional, batch_first=True, dropout=self.dropout)
        self.h5 = nn.Linear(in_features=self.hidden_size * self.D, out_features=self.n_joint * 3)
        self.relu = nn.ReLU()
    
    def forward(self, wav): # wav : [batch, longest_seq + 2 * context, mfcc channel]
        batch_size = wav.size(0)
        longest_seq = wav.size(1)
        
        wav = wav.permute(0, 2, 1) # [batch, mfcc channel, longest_seq]

        h1 = self.h1(wav).permute(0, 2, 1) # [batch, longest_seq, mfcc channel]
        h2 = self.h2(h1) 
        h3 = self.h3(h2)
        output, (hidden, cell) = self.h4(h3) # output : [batch, longest_seq, D * hidden]
                                             # hidden : [D * num_layer, batch, hidden]
                                             # cell : [D * num_layer, batch, hidden]
        h5 = self.h5(self.relu(output)) # [batch, longest_seq, n_joint * 3]
        
        return h5  
