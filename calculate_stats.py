import os
import numpy as np
import glob

# ================= CONFIGURATION =================
# Path to the raw .npy files
DATASET_PATH = "preprocessed"
OUTPUT_PATH = "preprocessed_ref"

# Motion dimension (26 joints * 3 coords = 78)
MOTION_DIM = 78 
# =================================================

def main():
    print(f"Scanning {DATASET_PATH} for motion files...")
    file_paths = glob.glob(os.path.join(DATASET_PATH, '*.npy'))
    
    if not file_paths:
        print(f"[Error] No .npy files found in {DATASET_PATH}")
        return

    valid_motion_frames = []
    skipped_files = 0

    for f in file_paths:
        try:
            data = np.load(f, allow_pickle=True)

            if len(data.shape) == 2 and data.shape[1] == MOTION_DIM:
                # prevent exceptions
                if np.isnan(data).any() or np.isinf(data).any():
                    print(f"Warning: Skipping {os.path.basename(f)} containing NaN/Inf")
                    continue
                
                valid_motion_frames.append(data)
            else:
                skipped_files += 1

        except Exception as e:
            print(f"Error reading {f}: {e}")

    print(f"Found {len(valid_motion_frames)} valid motion sequences.")
    print(f"Skipped {skipped_files} other files (Audio/Hierarchy).")

    if not valid_motion_frames:
        print("[Error] No valid motion files found: Check your paths.")
        return

    # Stack all frames to calculate global stats
    print("Concatenating data...")
    full_data = np.concatenate(valid_motion_frames, axis=0) 
    print(f"Total Frames: {full_data.shape[0]}")

    print("Calculating statistics...")
    motion_mean = np.mean(full_data, axis=0)
    motion_std = np.std(full_data, axis=0)

    # Prevent Division by Zero
    epsilon = 1e-5
    near_zero_idx = motion_std < epsilon
    
    # Replace tiny stds with 1.0
    motion_std[near_zero_idx] = 1.0
    
    print(f"Fixed {np.sum(near_zero_idx)} joints that had zero variance.")

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    np.save(os.path.join(OUTPUT_PATH, 'motion_mean.npy'), motion_mean)
    np.save(os.path.join(OUTPUT_PATH, 'motion_std.npy'), motion_std)

    print("------------------------------------------------")
    print("Success! Stats saved.")
    print(f"Mean Path: {os.path.join(OUTPUT_PATH, 'motion_mean.npy')}")
    print(f"Std Path:  {os.path.join(OUTPUT_PATH, 'motion_std.npy')}")
    print("------------------------------------------------")
    
    # Validation
    print("Sanity Check:")
    print(f"Mean (First 5): {motion_mean[:5]}")
    print(f"Std  (First 5): {motion_std[:5]}")
    print("Std should not be extremely small (e.g. 1e-6) or huge (e.g. 1000)")

if __name__ == "__main__":
    main()