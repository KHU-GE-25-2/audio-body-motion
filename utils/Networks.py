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
    
class ResidualBlock(nn.Module):
    """
    A simple Residual Block: x = x + MLP(x)
    Helps smooth out the signal and prevents gradient vanishing.
    """
    def __init__(self, hidden_size, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x):
        # The "Skip Connection": Input (x) + Processed (block(x))
        return x + self.block(x)

class AudioGestureLSTMRevised(nn.Module):
    def __init__(self, mfcc_channel, context, hidden_size, n_joint, silence_npy_path=None, device='cpu', dropout=0.1, num_layers=2, bidirectional=True):
        super(AudioGestureLSTMRevised, self).__init__()
        
        self.device = device
        input_size = mfcc_channel # 26
        output_size = n_joint     # 78
        
        # Audio Encoder (Pre-Net)
        self.audio_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Recurrent Core
        self.lstm = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Calculate LSTM output size (doubles if bidirectional)
        lstm_out_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Projection to bring bidirectional output back to hidden_size
        self.post_lstm_proj = nn.Linear(lstm_out_size, hidden_size)

        # --- 3. Residual Correction (The Smoother) ---
        # Allows the model to refine the flow
        self.res_block = ResidualBlock(hidden_size, dropout)
        
        # --- 4. Motion Decoder (Post-Net) ---
        self.output_decoder = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # x shape: (Batch, Time, 26)
        x = self.audio_encoder(x) 
        x, _ = self.lstm(x)
        x = self.post_lstm_proj(x)
        x = self.res_block(x)
        out = self.output_decoder(x)
        
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
