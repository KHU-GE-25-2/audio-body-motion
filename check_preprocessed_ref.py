import numpy as np
import os

target_folder = 'preprocessed_ref'

print(f"--- INSPECTING {target_folder} ---")

try:
    std_path = os.path.join(target_folder, 'motion_std.npy')
    mean_path = os.path.join(target_folder, 'motion_mean.npy')
    
    if not os.path.exists(std_path):
        print("ERROR: File not found!")
        exit()

    std = np.load(std_path)
    mean = np.load(mean_path)

    print(f"Loaded Std Shape: {std.shape}")
    
    # Index 0 (Root X - Should be 1.0)
    print(f"Index 0 (Root): {std[0]}")
    
    # Index 3 (Spine/Hip Rotation
    # should be small like 0.2)
    val_3 = std[3]
    print(f"Index 3 (Body): {val_3}")

    if val_3 == 1.0:
        print("\nDIAGNOSIS: The stats file is BROKEN.")
        print("   It contains 1.0 for moving joints. Normalization will NOT work.")
        print("   Action: Re-run calculate_stats.py.")
    else:
        print("\nDIAGNOSIS: The stats file is VALID.")
        print("   If training is still failing, the issue is in dataset.py __getitem__.")

except Exception as e:
    print(f"Error: {e}")