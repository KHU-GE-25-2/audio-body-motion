import os 
import cv2 
import torch 
import shutil
import numpy as np
import moviepy.editor as mpe
import re
from utils.feature_tools import * 


def get_save_path(results_path='models'):
    idx = max([int(f.split(f'_')[-1]) for f in os.listdir(results_path) if f.startswith('train')]+[0]) + 1 
    path = os.path.join(os.path.abspath(results_path), f'train_{idx}')
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def get_numpy(np_path) : 
    return np.load(np_path)

def get_inference_input(input_path, mfcc_channel, silence_npy_path, context):
    input_wav = get_wav_vector(input_path, mfcc_channel)
    print(f'Load input wav file, has length {input_wav.shape[0]}')

    wav_pad = get_numpy(silence_npy_path)
    wav_pad = np.repeat(wav_pad, repeats=context, axis=0)
    
    input_wav = np.append(wav_pad, input_wav, axis=0)
    input_wav = np.append(input_wav, wav_pad, axis=0)

    return torch.tensor(input_wav).unsqueeze(0).float() # [1(batch), 2*context + total_time_step, mfcc_channel]

def load_video_saver(save_video_path, bg_size, fps):
    # return save_video instance 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = fps
    w, h = bg_size, bg_size
    save_video = cv2.VideoWriter(save_video_path, fourcc, fps, (w, h))
    return save_video

def rearrange_joint_line(joint_coor):
    joints = [] 
    coordinate = []
    for idx, coo in enumerate(joint_coor): 
        coordinate.append(float(coo))
        if (idx+1) % 3 == 0 : 
            joints.append(coordinate)
            coordinate = []
    return joints

def combine_video_and_audio(video_path, audio_path):
    print(f'add audio {audio_path} to video {video_path}...')
    origin_vid_path = video_path
    backup_vid_path = video_path.split('.')[0] + 'back.' + video_path.split('.')[-1]
    shutil.copy(origin_vid_path, backup_vid_path)

    my_clip = mpe.VideoFileClip(backup_vid_path)
    audio_background = mpe.AudioFileClip(audio_path)
    final_audio = mpe.CompositeAudioClip([audio_background])
    final_clip = my_clip.set_audio(final_audio)
    final_clip.write_videofile(origin_vid_path, audio_codec="aac")

    os.remove(backup_vid_path)

def draw_gesture_and_save_video(outputs, child_list, output_path, bg_size, size_mag, fps):
    print(f'start to draw gesture')
    print(f'total frame of output results : {len(outputs)}')
    outputs = outputs.tolist()

    save_video = load_video_saver(output_path, bg_size, fps)

    x_offset = bg_size * (1/2)
    y_offset = bg_size * (2/3) 

    for output in outputs:
        joints = rearrange_joint_line(output)

        # make empty image
        img = np.zeros([bg_size, bg_size, 3],dtype=np.uint8)
        img.fill(255)

        # get x, y and draw! 
        for jnt_idx, joint in enumerate(joints): 
            x = int((joint[0] * size_mag) + x_offset)
            y = int(y_offset - (joint[1] * size_mag))
            
            # draw circle
            img = cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
            
            # draw line
            for child_idx in child_list[jnt_idx] : 
                target_x = int(joints[child_idx][0] * size_mag + x_offset)
                target_y = int(y_offset - joints[child_idx][1] * size_mag)
                img = cv2.line(img, (x, y), (target_x, target_y), (0, 0, 180), 1)
        
        save_video.write(img)

    print(f'Done to make gesture video')
    save_video.release()

def save_bvh(save_path, motion_data, ref_bvh_path):
    """
    Saves model output (Absolute Positions) to a BVH file.
    It modifies the header to accept POSITION channels for all joints.
    """
    print(f"Saving Position-Based BVH to {save_path}...")
    
    # 1. Read and Modify Header
    header_lines = []
    with open(ref_bvh_path, 'r') as f:
        for line in f:
            if "MOTION" in line:
                break
            
            # THE MAGIC TRICK:
            # Change "CHANNELS 3 Zrotation Xrotation Yrotation"
            # To     "CHANNELS 3 Xposition Yposition Zposition"
            if "CHANNELS 3" in line:
                line = re.sub(r"Zrotation Xrotation Yrotation", "Xposition Yposition Zposition", line)
                # Also handle variations like "Xrotation Yrotation Zrotation"
                line = re.sub(r"[XYZ]rotation [XYZ]rotation [XYZ]rotation", "Xposition Yposition Zposition", line)
            
            header_lines.append(line)
    
    # 2. Inspect Data Dimensions
    frames, data_channels = motion_data.shape
    
    # Count how many channels the NEW header expects
    header_channels = 0
    for line in header_lines:
        if "CHANNELS" in line:
            parts = line.strip().split()
            header_channels += int(parts[1])

    # 3. Handle End Sites (Truncate)
    diff = data_channels - header_channels
    if diff > 0:
        print(f"⚠️ Truncating {diff} extra channels (End Sites).")
        motion_data = motion_data[:, :header_channels]
    
    # 4. Write File
    with open(save_path, 'w') as f:
        f.writelines(header_lines)
        f.write("MOTION\n")
        f.write(f"Frames: {frames}\n")
        f.write("Frame Time: 0.050000\n") 
        
        for frame in motion_data:
            line = " ".join(f"{x:.6f}" for x in frame)
            f.write(line + "\n")
            
    print(f"✅ Saved Position-BVH to {save_path}")