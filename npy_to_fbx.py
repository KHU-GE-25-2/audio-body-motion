import bpy
import numpy as np
import os
from dotenv import load_dotenv

# ================= CONFIGURATION =================
load_dotenv()

BVH_PATH = os.getenv("BVH_PATH")
NPY_PATH = os.getenv("NPY_PATH")
FBX_OUTPUT = os.getenv("FBX_OUTPUT")

# force the character to be TARGET_HEIGHT_METERS tall (UE5 Skeleton is usually 1.8m tall)
TARGET_HEIGHT_METERS = 1.75

# AXIS MODE: 1 = Standard (Y-Up Data to Z-Up Blender)
AXIS_MODE = 1
TARGET_FPS = 30 
FRAME_TIME = 1.0 / TARGET_FPS 
# =================================================

def get_bvh_node_order(bvh_path):
    """Parses BVH to get the EXACT order of data points (Bones + End Sites)."""
    nodes = []
    stack = []
    with open(bvh_path, 'r') as f:
        for line in f:
            if "MOTION" in line: break
            parts = line.strip().split()
            if not parts: continue
            token = parts[0]
            if token == "ROOT" or token == "JOINT":
                name = parts[1]
                nodes.append(name)
                stack.append(name)
            elif token == "End":
                parent = stack[-1]
                nodes.append(f"EndSite_{parent}") 
                stack.append(f"EndSite_{parent}")
            elif token == "}":
                if stack: stack.pop()
    print(f"BVH Parser found {len(nodes)} data points.")
    return nodes

def ensure_clean_bvh(input_path):
    temp_path = input_path.replace(".bvh", "_clean_temp.bvh")
    header = []
    channel_count = 0
    with open(input_path, 'r') as f:
        for line in f:
            if "MOTION" in line: break
            header.append(line)
            if "CHANNELS" in line:
                channel_count += int(line.strip().split()[1])
    with open(temp_path, 'w') as f:
        f.writelines(header)
        f.write("MOTION\nFrames: 1\n")
        f.write(f"Frame Time: {FRAME_TIME:.6f}\n") 
        f.write(" ".join(["0.0"] * channel_count) + "\n")
    return temp_path

def calculate_auto_scale(data):
    # Get Frame 0 Data
    frame0 = data[0]
    
    # Reshape to (N_Joints, 3) points
    n_points = len(frame0) // 3
    points = frame0[:n_points*3].reshape(-1, 3)
    
    # Calculate Extents (Max - Min) for X, Y, Z
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    extents = max_vals - min_vals
    
    # Find the largest dimension (usually the height)
    raw_height = np.max(extents)
    
    if raw_height < 0.001: 
        print("Warning: Data seems flat or zero. Defaulting scale to 0.01")
        return 0.01
        
    scale_factor = TARGET_HEIGHT_METERS / raw_height
    
    print(f"[Auto-Scale] Raw Data Height: {raw_height:.2f} units")
    print(f"[Auto-Scale] Target Height:   {TARGET_HEIGHT_METERS:.2f} meters")
    print(f"[Auto-Scale] Calculated Factor: {scale_factor:.5f}")
    
    return scale_factor

def process_and_export():
    scene = bpy.context.scene
    scene.render.fps = TARGET_FPS
    
    # Import Skeleton
    clean_bvh = ensure_clean_bvh(BVH_PATH)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    print("Importing Skeleton...")
    bpy.ops.import_anim.bvh(filepath=clean_bvh, axis_up='Y', axis_forward='-Z')
    if os.path.exists(clean_bvh): os.remove(clean_bvh)
    
    armature = bpy.context.object
    armature.name = "AI_Skeleton"

    # Get Data Order
    npy_order = get_bvh_node_order(BVH_PATH)

    # Disconnect Bones
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    for bone in armature.data.edit_bones:
        bone.use_connect = False
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Add Dummy Mesh
    # to fix Unreal Engine import error due to mesh absence
    bpy.ops.mesh.primitive_cube_add(size=0.1, location=(0, 0, 0))
    dummy_mesh = bpy.context.object
    dummy_mesh.name = "Dummy_SK_Mesh"
    bpy.ops.object.select_all(action='DESELECT')
    dummy_mesh.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    if dummy_mesh.animation_data: dummy_mesh.animation_data_clear()

    print(f"Loading NPY Data: {NPY_PATH}")
    data = np.load(NPY_PATH)

    GLOBAL_SCALE = calculate_auto_scale(data)
    
    # Setup Empties
    bpy.ops.object.mode_set(mode='OBJECT')
    index_to_empty = {} 
    
    for i, name in enumerate(npy_order):
        if name in armature.pose.bones:
            bpy.ops.object.empty_add(type='PLAIN_AXES', radius=0.1)
            emp = bpy.context.object
            emp.name = f"Ctrl_{name}"
            index_to_empty[i] = emp
            
            pbone = armature.pose.bones[name]
            bpy.context.view_layer.objects.active = armature
            bpy.ops.object.mode_set(mode='POSE')
            c = pbone.constraints.new('COPY_LOCATION')
            c.target = emp
            c.owner_space = 'WORLD'
            c.target_space = 'WORLD'
            bpy.ops.object.mode_set(mode='OBJECT')

    # Bake Loop
    print(f"Baking {data.shape[0]} frames...")
    scene.frame_start = 0
    scene.frame_end = data.shape[0]
    
    for frame_idx, frame_data in enumerate(data):
        target_frame = frame_idx + 1
        scene.frame_set(target_frame)
        
        for i, emp in index_to_empty.items():
            if i*3 + 2 >= len(frame_data): break
            
            raw_x = frame_data[i*3]
            raw_y = frame_data[i*3+1]
            raw_z = frame_data[i*3+2]
            
            # AXIS SWAP: Y-Up (Data) to Z-Up (Blender)
            if AXIS_MODE == 1:
                new_x = raw_x * GLOBAL_SCALE
                new_y = -raw_z * GLOBAL_SCALE 
                new_z = raw_y * GLOBAL_SCALE
                emp.location = (new_x, new_y, new_z)
            
        bpy.context.view_layer.update()
        
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='POSE')
        for pbone in armature.pose.bones:
            pbone.keyframe_insert(data_path="location", options={'INSERTKEY_VISUAL'})
        bpy.ops.object.mode_set(mode='OBJECT')
        
        if frame_idx % 50 == 0: print(f"Processing {frame_idx}...")

    # Cleanup & Export
    bpy.ops.object.select_all(action='DESELECT')
    for idx, emp in index_to_empty.items(): emp.select_set(True)
    bpy.ops.object.delete()
    
    print(f"Exporting to {FBX_OUTPUT}...")
    bpy.ops.object.select_all(action='SELECT')
    
    bpy.ops.export_scene.fbx(
        filepath=FBX_OUTPUT, 
        use_selection=True, 
        add_leaf_bones=False, 
        bake_anim=True,
        object_types={'ARMATURE', 'MESH'}, 
        bake_anim_use_nla_strips=False,    
        bake_anim_use_all_actions=False,
        mesh_smooth_type='FACE',
        axis_forward='-Z',
        axis_up='Y'
    )
    print("Export Complete.")

if __name__ == "__main__":
    process_and_export()