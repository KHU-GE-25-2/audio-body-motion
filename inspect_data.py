import numpy as np
import matplotlib.pyplot as plt
import os
import glob

DATA_PATH = "preprocessed" # Your path
HARD_LIMIT = 360.0

def inspect():
    files = sorted(glob.glob(os.path.join(DATA_PATH, '*Y.npy')))
    
    print(f"Checking {len(files)} files for outliers...")
    
    bad_files = []
    
    for f in files:
        data = np.load(f)
        # Check raw max value
        if np.isnan(data).any():
            print(f"❌ Found NaN values in {file_path}")

        # 3. Check for massive spikes (actual glitches)
        elif np.max(np.abs(data)) > HARD_LIMIT:
            print(f"⚠️ Suspiciously high value ({np.max(np.abs(data)):.2f}) in {file_path}")
            bad_files.append((f, np.max(np.abs(data))))

        else:
            # If it passes these, the file is likely fine
            pass

    print(f"Found {len(bad_files)} suspicious files.")
    
    if len(bad_files) > 0:
        # Plot the first bad one
        bad_file, val = bad_files[0]
        print(f"Plotting bad file: {bad_file} (Max: {val})")
        
        data = np.load(bad_file)
        plt.figure(figsize=(10, 5))
        plt.plot(data)
        plt.title(f"Visualizing Glitches in {os.path.basename(bad_file)}")
        plt.xlabel("Time")
        plt.ylabel("Joint Value")
        plt.show()

def find_unit():
    # Load the specific file mentioned in your log
    file_path = r"preprocessed\MM_M_C_F_C_S064_001_Y.npy"
    data = np.load(file_path)

    print(f"Shape: {data.shape}")
    print(f"Min Value: {np.min(data):.4f}")
    print(f"Max Value: {np.max(data):.4f}")
    print(f"Mean: {np.mean(data):.4f}")

    # Check if it looks like degrees (values > 3.14)
    if np.max(np.abs(data)) > 3.14159 and np.max(np.abs(data)) <= 180:
        print(">>> Diagnosis: Data is likely in DEGREES.")
    elif np.max(np.abs(data)) > 180:
        print(">>> Diagnosis: Data might be Position (cm) or unnormalized indices.")

if __name__ == "__main__":
    inspect()
    # find_unit()