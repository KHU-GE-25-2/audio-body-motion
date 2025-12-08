import re

def check_bvh_channels(bvh_path, model_dim=78):
    print(f"Inspecting Header: {bvh_path}")
    expected_channels = 0
    
    with open(bvh_path, 'r') as f:
        for line in f:
            if "MOTION" in line:
                break
            if "CHANNELS" in line:
                # Line looks like: CHANNELS 3 Zrotation Xrotation Yrotation
                parts = line.strip().split()
                count = int(parts[1])
                expected_channels += count
                
    print(f"--- RESULTS ---")
    print(f"BVH Header Expects: {expected_channels} columns")
    print(f"Model Output Has:   {model_dim} columns")
    
    diff = expected_channels - model_dim
    if diff == 3:
        print(f"⚠️  MISMATCH FOUND: Header wants {diff} more values.")
        print(">>> CAUSE: Your model predicts Rotations, but the BVH Header also wants Root Position (X,Y,Z).")
        print(">>> FIX: We need to pad the data with '0 0 0' for the root position.")
    elif diff != 0:
        print(f"⚠️  MISMATCH: Difference is {diff}. Skeleton structures do not match.")
    else:
        print("✅ Match! The issue might be Rotation Order (e.g. XYZ vs ZXY).")

# --- Run it on your files ---
# Update these paths to yours
check_bvh_channels("ref_data_folder/hierarchy.bvh", model_dim=78)