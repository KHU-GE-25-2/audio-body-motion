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
    def __init__(self, mfcc_channel, context, hidden_size, n_joint, silence_npy_path, device, dropout=0.2, num_layers=2, bidirectional=True):
        super().__init__()
        self.mfcc_channel = mfcc_channel
        self.context = context
        self.hidden_size = hidden_size
        self.n_joint = n_joint
        self.device = device

        self.D = 2 if bidirectional else 1 # For bidirectional LSTM

        # --- 1. Initial Temporal Convolution ---
        # This layer expands the receptive field for each time step.
        self.temporal_conv = nn.Conv1d(
            in_channels=self.mfcc_channel,
            out_channels=self.hidden_size, # Project to hidden_size early
            kernel_size=(2 * self.context + 1),
            stride=1
        )
        self.conv_norm = nn.LayerNorm(self.hidden_size)

        # --- 2. Pre-LSTM Processing Blocks with Residual Connection ---
        # This deepens the feature extraction before the sequential analysis.
        self.pre_lstm_block = nn.Sequential(
            AudioProcessorBlock(self.hidden_size, self.hidden_size, dropout=dropout),
            AudioProcessorBlock(self.hidden_size, self.hidden_size, dropout=dropout)
        )

        # --- 3. Core Sequential Model ---
        self.lstm = nn.LSTM(
            input_size=self.hidden_size, # Input is now the processed features
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # --- 4. Output Projection Head ---
        self.output_head = nn.Linear(self.hidden_size * self.D, self.n_joint * 3)

    def forward(self, wav): # wav shape: [batch, seq_len + 2 * context, mfcc_channel]
        # Permute for Conv1d: [batch, mfcc_channel, seq_len + 2*context]
        wav = wav.permute(0, 2, 1)

        # --- Stage 1: Temporal Feature Extraction ---
        conv_out = self.temporal_conv(wav)
        # Permute back for sequence processing: [batch, seq_len, hidden_size]
        conv_out = conv_out.permute(0, 2, 1)
        # Apply normalization
        x = self.conv_norm(conv_out)

        # --- Stage 2: Deep Feature Processing with Residual Connection ---
        # The input 'x' is added to the output of the block, creating a skip connection.
        processed_x = self.pre_lstm_block(x) + x

        # --- Stage 3: Sequential Analysis ---
        lstm_output, _ = self.lstm(processed_x)

        # --- Stage 4: Final Projection ---
        # No activation here, as output coordinates can be positive or negative.
        output_gestures = self.output_head(lstm_output)

        return output_gestures

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
