import os 
import argparse
import torch 
import numpy as np

from utils.utils import (
    get_inference_input,
    get_child_list_from_bvh,
    draw_gesture_and_save_video,
    combine_video_and_audio,
)

from utils.Networks import AudioGestureLSTM, AudioGestureLSTMRevised

def parse_args():
    parser = argparse.ArgumentParser()
    # IO info
    parser.add_argument('-m', '--model_path', type=str, help='trained model path')
    parser.add_argument('-i', '--input_wav', type=str, help='input wav file')
    parser.add_argument('--hierarchy_bvh_path', type=str, help='sample bvh file for getting joint hierarchy info')
    
    parser.add_argument('--stats_dir', type=str, default='preprocessed_norm', help='folder containing motion_mean.npy and motion_std.npy')
    
    parser.add_argument('--silence_npy_path', type=str, default='reference_data/silence.npy', help='silence npy file path')
    parser.add_argument('--output_path', type=str, default='inferenced.mp4', help='path for output video in mp4 format')
    
    ## Model setting. same as train 
    parser.add_argument('--mfcc_channel', type=int, default=26, help='wav mfcc channel')
    parser.add_argument('--context', type=int, default=30, help='''one way context value added along with input t, 
                                                                   total input : context * 2 + 1''')
    parser.add_argument('--hidden_size', type=int, default=256, help='LSTM hidden size')
    parser.add_argument('--dropout', type=float, default=0, help='LSTM dropout rate')
    parser.add_argument('--n_joint', type=int, default=78, help='num of motion keypoint')

    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--bidirectional", dest="bidirectional", action="store_true")
    parser.add_argument("--no-bidirectional", dest="bidirectional", action="store_false")
    parser.set_defaults(bidirectional=True)

    ## Draw setting 
    parser.add_argument('--bg_size', type=int, default=512, help='Size of output video')
    parser.add_argument('--size_mag', type=float, default=2.0, help='scaling factor to size up or down output skeleton')
    parser.add_argument('--fps', type=int, default=20, help='FPS for output video')

    parser.add_argument('--motion_loudness', type=float, default=1.0, help='Motion Gain: 1.0=Normal, 1.2=Energetic, 0.8=Subtle')
    
    parser.add_argument('--revised_model', action="store_true", help='use the revised model version')

    return parser.parse_args()

def main():
    args = parse_args()
    
    assert os.path.exists(args.model_path), f'Check your trained model ckpt file path.. {args.model_path}'
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('target device :', device)

    model = None

    if (args.revised_model):
        print("Revised Model running")
        model = AudioGestureLSTMRevised(
            args.mfcc_channel,
            args.context,
            args.hidden_size,
            args.n_joint,
            args.silence_npy_path,
            device,
            dropout=args.dropout
        ).to(device)
    else:
        print("Old Model running")
        model = AudioGestureLSTM(
            args.mfcc_channel, 
            args.context, 
            args.hidden_size, 
            args.n_joint, 
            args.silence_npy_path, 
            device
        ).to(device)

    if torch.cuda.is_available() :
        model.load_state_dict(torch.load(args.model_path))
    else : 
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))

    model.eval() # Important: Set model to eval mode!

    input_wav = get_inference_input(args.input_wav, args.mfcc_channel, args.silence_npy_path, args.context).to(device)

    with torch.no_grad():
        output = model(input_wav) # [1(batch), total_time_step, n_joint * 3]

        if args.motion_loudness != 1.0:
            print(f"⚡ Applying Motion Loudness: x{args.motion_loudness}")
            output = output * args.motion_loudness

        print(f"Raw Model Output Range: [{output.min():.3f}, {output.max():.3f}] (Should be approx -1 to 1)")

    # ==========================================
    # START: DE-NORMALIZATION (The Fix)
    # ==========================================
    print("De-normalizing output...")

    stats_path = os.path.join(args.stats_dir, 'stats_Y.npz')

    if os.path.exists(stats_path):
        print(f"Found stats file: {stats_path}")
        stats = np.load(stats_path)
        
        # Extract the Min and Range
        # We used keys 'min', 'max', 'range' in the normalization script
        # If the keys are different, check your npz file (e.g., 'data_min')
        try:
            # Try new keys first
            raw_min = stats['min']
            raw_range = stats['range']
        except:
            # Fallback to old keys if any
            raw_min = stats['data_min']
            raw_range = stats['data_range']

        # Convert to Tensor
        motion_min = torch.tensor(raw_min).float().to(device)
        motion_range = torch.tensor(raw_range).float().to(device)
        
        # --- THE MATH ---
        # Normalized = 2 * (x - min) / range - 1
        # Inverse: x = ((Normalized + 1) / 2) * range + min
        
        output = ((output + 1.0) / 2.0) * motion_range + motion_min
        
        print(f"Restored Output Range: [{output.min():.3f}, {output.max():.3f}] (Should be ~ -180 to 180)")
    else:
        print(f"⚠️ WARNING: stats_Y.npz NOT FOUND in {args.stats_dir}")
        print("Using RAW output (Animation will look broken!)")
    # ==========================================
    # END: DE-NORMALIZATION
    # ==========================================
    # else:
         
    #     output = model(input_wav) # [1(batch), total_time_step, n_joint * 3]

    child_list = get_child_list_from_bvh(args.hierarchy_bvh_path)
    draw_gesture_and_save_video(output.squeeze(0), child_list, args.output_path, 
                                args.bg_size, args.size_mag, args.fps)
    combine_video_and_audio(args.output_path, args.input_wav)

if __name__=='__main__':
    main()