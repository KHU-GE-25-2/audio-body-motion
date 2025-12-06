import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr

# Assumes you have your class definitions here
from utils.Networks import AudioGestureLSTM, AudioGestureLSTMRevised

def calculate_jerk(motion_data):
    """
    Calculates the 'smoothness' or jitter of motion.
    Jerk is the 3rd derivative of position (change in acceleration).
    motion_data: (Time, Joints)
    """
    # 1st Derivative: Velocity (pos[t+1] - pos[t])
    velocity = np.diff(motion_data, axis=0)
    # 2nd Derivative: Acceleration
    acceleration = np.diff(velocity, axis=0)
    # 3rd Derivative: Jerk
    jerk = np.diff(acceleration, axis=0)
    
    # Average magnitude of jerk across all frames and joints
    return np.mean(np.abs(jerk))

def calculate_correlation(audio_data, motion_data):
    """
    Calculates correlation between Audio Loudness and Motion Velocity.
    """
    # 1. Motion Velocity (How fast are we moving?)
    # Shape: (Time-1, Joints) -> Average over joints -> (Time-1,)
    velocity = np.abs(np.diff(motion_data, axis=0)).mean(axis=1)
    
    # 2. Audio Loudness (Approximate from MFCC)
    # MFCC 0 is usually the "Energy" or loudness coefficient
    # We clip audio to match velocity length
    audio_envelope = audio_data[:len(velocity), 0] 
    
    # 3. Pearson Correlation
    # Handle edge case of constant output (std=0)
    if np.std(velocity) < 1e-6 or np.std(audio_envelope) < 1e-6:
        return 0.0
        
    corr, _ = pearsonr(audio_envelope, velocity)
    return corr

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to .ckpt file')
    parser.add_argument('--test_dir', type=str, default='preprocessed_norm', help='Folder with _X.npy and _Y.npy files')
    parser.add_argument('--stats_dir', type=str, default='preprocessed_norm', help='Folder with stats_Y.npz')
    parser.add_argument('--silence_npy_path', type=str, default='reference_data/silence.npy', help='silence npy file path')

    # Model Config (Must match training!)
    parser.add_argument('--mfcc_channel', type=int, default=26)
    parser.add_argument('--n_joint', type=int, default=78)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--context', type=int, default=30)
    parser.add_argument('--revised_model', action="store_true", help="Use V2 Architecture")
    parser.add_argument('--energy', type=float, default=1.0, help="Inference energy multiplier")

    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluator running on: {device}")

    # --- 1. Load Model ---
    model = None
    if args.revised_model:
        model = AudioGestureLSTMRevised(args.mfcc_channel, args.context, args.hidden_size, args.n_joint, device=device, num_layers=2, bidirectional=True)
    else:
        model = AudioGestureLSTM(
            args.mfcc_channel, 
            args.context, 
            args.hidden_size, 
            args.n_joint, 
            args.silence_npy_path,
            device=device)
        
    # Load Weights
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # --- 2. Load Stats for De-Normalization ---
    stats_path = os.path.join(args.stats_dir, 'stats_Y.npz')
    if not os.path.exists(stats_path):
        print("❌ Error: Stats file not found. Cannot evaluate metrics in Real Scale.")
        return
        
    stats = np.load(stats_path)
    # Handle different key naming conventions
    try:
        raw_min = stats['min']
        raw_range = stats['range']
    except:
        raw_min = stats['data_min']
        raw_range = stats['data_range']
        
    data_min = torch.tensor(raw_min).float().to(device)
    data_range = torch.tensor(raw_range).float().to(device)

    # --- 3. Find Test Files ---
    # We look for paired X (Audio) and Y (Motion) files
    x_files = sorted(glob.glob(os.path.join(args.test_dir, "*_X.npy")))
    print(f"Found {len(x_files)} test files. Starting evaluation...")

    # Store scores
    mae_scores = []
    jerk_real_scores = []
    jerk_gen_scores = []
    corr_real_scores = []
    corr_gen_scores = []

    with torch.no_grad():
        for x_path in x_files:
            y_path = x_path.replace("_X.npy", "_Y.npy")
            if not os.path.exists(y_path): continue

            # Load Data (Normalized)
            input_wav = np.load(x_path).astype(np.float32) # (Time, 26)
            gt_motion = np.load(y_path).astype(np.float32) # (Time, 78)
            
            # Clip to match length
            min_len = min(len(input_wav), len(gt_motion))
            input_wav = input_wav[:min_len]
            gt_motion = gt_motion[:min_len]

            # Prepare Batch for Model
            # Input shape needs to be (1, Time, 26)
            inp_tensor = torch.tensor(input_wav).unsqueeze(0).to(device)
            
            # --- INFERENCE ---
            gen_motion_tensor = model(inp_tensor) # (1, Time, 78)
            
            # Apply Energy Multiplier (optional)
            if args.energy != 1.0:
                gen_motion_tensor = gen_motion_tensor * args.energy

            # --- DE-NORMALIZE ---
            # We want metrics in Real Degrees, not -1 to 1
            # Inverse: x = ((Norm + 1) / 2) * Range + Min
            gen_motion = ((gen_motion_tensor.squeeze(0) + 1.0) / 2.0) * data_range + data_min
            
            # We must also de-normalize the Ground Truth for fair comparison
            gt_motion_tensor = torch.tensor(gt_motion).to(device)
            gt_motion_real = ((gt_motion_tensor + 1.0) / 2.0) * data_range + data_min

            # Convert to CPU Numpy for Calc
            gen_np = gen_motion.cpu().numpy()
            gt_np = gt_motion_real.cpu().numpy()
            
            # --- CALCULATE METRICS ---

            # [FIX] Force lengths to match exactly before comparison
            final_len = min(len(gen_np), len(gt_np))
            gen_np = gen_np[:final_len]
            gt_np = gt_np[:final_len]
            # [END FIX]
            
            # 1. Similarity (L1 Loss)
            mae = np.mean(np.abs(gen_np - gt_np))
            mae_scores.append(mae)
            
            # 2. Jitter (Smoothness)
            j_gen = calculate_jerk(gen_np)
            j_real = calculate_jerk(gt_np)
            jerk_gen_scores.append(j_gen)
            jerk_real_scores.append(j_real)
            
            # 3. Audio-Motion Correlation
            c_gen = calculate_correlation(input_wav, gen_np)
            c_real = calculate_correlation(input_wav, gt_np)
            corr_gen_scores.append(c_gen)
            corr_real_scores.append(c_real)
            
            print(f"Processed {os.path.basename(x_path)} | MAE: {mae:.2f}")

    # --- 4. Final Report ---
    print("\n" + "="*40)
    print("      EVALUATION RESULTS      ")
    print("="*40)
    print(f"Avg L1 Error (Similarity): {np.mean(mae_scores):.4f} (Lower is better)")
    print("-" * 40)
    print(f"Real Data Jitter:    {np.mean(jerk_real_scores):.4f}")
    print(f"Generated Jitter:    {np.mean(jerk_gen_scores):.4f}")
    
    jitter_ratio = np.mean(jerk_gen_scores) / np.mean(jerk_real_scores)
    if jitter_ratio > 1.1:
        print(f"⚠️  Verdict: Generated motion is JITTERY (Ratio: {jitter_ratio:.2f})")
    elif jitter_ratio < 0.8:
        print(f"⚠️  Verdict: Generated motion is OVER-SMOOTHED (Ratio: {jitter_ratio:.2f})")
    else:
        print(f"✅ Verdict: Smoothness matches Reality (Ratio: {jitter_ratio:.2f})")
        
    print("-" * 40)
    print(f"Real Data Audio-Motion Corr: {np.mean(corr_real_scores):.4f}")
    print(f"Generated Audio-Motion Corr: {np.mean(corr_gen_scores):.4f}")
    
    if np.mean(corr_gen_scores) > np.mean(corr_real_scores):
        print("✅ Verdict: Model reacts strongly to the beat.")
    else:
        print("ℹ️  Verdict: Model is more passive than the real actor.")
    print("="*40)

if __name__ == '__main__':
    main()