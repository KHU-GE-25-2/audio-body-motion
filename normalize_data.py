import numpy as np
import os
import glob

def normalize_split_dataset(input_dir, output_dir):
    # --- 1. Setup ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Gather all files
    all_files = glob.glob(os.path.join(input_dir, "*.npy"))
    
    # Separate into X (Input) and Y (Output) lists based on filename
    x_files = [f for f in all_files if "_X.npy" in f]
    y_files = [f for f in all_files if "_Y.npy" in f]
    
    if not x_files or not y_files:
        print("❌ Error: Could not find both _X and _Y files in the folder.")
        print(f"Found {len(x_files)} X files and {len(y_files)} Y files.")
        return

    print(f"Found {len(x_files)} Input (X) files and {len(y_files)} Output (Y) files.")

    # ==========================================
    # PART A: Process Inputs (X) - 26 Dimensions
    # ==========================================
    print("\n--- Processing Inputs (X) ---")
    x_data_list = []
    for fp in x_files:
        x_data_list.append(np.load(fp))
        
    # Stack to find global stats for X
    full_X = np.vstack(x_data_list)
    min_X = np.min(full_X, axis=0)
    max_X = np.max(full_X, axis=0)
    range_X = max_X - min_X
    range_X[range_X == 0] = 1.0 # Safety
    
    print(f"Global Max X: {np.max(full_X):.2f}")
    
    # Normalize and Save X files
    for fp in x_files:
        data = np.load(fp)
        norm_data = 2 * (data - min_X) / range_X - 1
        
        base_name = os.path.basename(fp)
        save_path = os.path.join(output_dir, base_name)
        np.save(save_path, norm_data)

    # ==========================================
    # PART B: Process Outputs (Y) - 78 Dimensions
    # ==========================================
    print("\n--- Processing Outputs (Y) ---")
    y_data_list = []
    for fp in y_files:
        y_data_list.append(np.load(fp))
        
    # Stack to find global stats for Y
    full_Y = np.vstack(y_data_list)
    min_Y = np.min(full_Y, axis=0)
    max_Y = np.max(full_Y, axis=0)
    range_Y = max_Y - min_Y
    range_Y[range_Y == 0] = 1.0 # Safety
    
    print(f"Global Max Y: {np.max(full_Y):.2f}")
    
    # Normalize and Save Y files
    for fp in y_files:
        data = np.load(fp)
        norm_data = 2 * (data - min_Y) / range_Y - 1
        
        base_name = os.path.basename(fp)
        save_path = os.path.join(output_dir, base_name)
        np.save(save_path, norm_data)

    # ==========================================
    # PART C: Save Stats Separately
    # ==========================================
    stats_path_X = os.path.join(output_dir, "stats_X.npz")
    stats_path_Y = os.path.join(output_dir, "stats_Y.npz")
    
    np.savez(stats_path_X, min=min_X, max=max_X, range=range_X)
    np.savez(stats_path_Y, min=min_Y, max=max_Y, range=range_Y)
    
    print("-" * 30)
    print("✅ Normalization Complete!")
    print(f"Saved normalized files to: {output_dir}")
    print(f"Saved Input Stats  -> {stats_path_X}")
    print(f"Saved Output Stats -> {stats_path_Y} (You need this for inference!)")

def normalize_padding_data():
    output_dir = "preprocessed_ref"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Generating normalized padding files in {output_dir}...")

    # 1. Create Silence (Audio Padding)
    # Shape: (1 frame, 26 features) -> Zeros
    silence = np.zeros((1, 26), dtype=np.float32)
    np.save(os.path.join(output_dir, "silence.npy"), silence)
    print("✅ Created silence.npy (26 dims)")

    # 2. Create Normalized Hierarchy (Motion Padding)
    # Shape: (1 frame, 78 features) -> Zeros
    # Since our data is -1 to 1, '0' represents the average/center pose, which is safe padding.
    hierarchy_norm = np.zeros((1, 78), dtype=np.float32)
    np.save(os.path.join(output_dir, "hierarchy_norm.npy"), hierarchy_norm)
    print("✅ Created hierarchy_norm.npy (78 dims)")

if __name__ == "__main__":
    INPUT_FOLDER = "preprocessed"
    OUTPUT_FOLDER = "preprocessed_norm"
    
    normalize_split_dataset(INPUT_FOLDER, OUTPUT_FOLDER)
    normalize_padding_data()