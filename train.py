import os
import tqdm
import argparse

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from utils.utils import *
from utils.dataset import AudioGestureDataset, AudioGestureDatasetRevised
from utils.collator import Collator
from utils.Networks import AudioGestureLSTM, AudioGestureLSTMRevised

# Assuming these imports exist in your project structure
# from utils.dataset import AudioGestureDataset, Collator, get_save_path
# from utils.Networks import AudioGestureLSTM, AudioGestureLSTMRevised

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset_path', type=str, default='preprocessed_norm', help='dataset folder contains train X and Y numpys')
    parser.add_argument('-r', '--results_path', type=str, default='models', help='trained results folder')
    parser.add_argument('--stats_dir', type=str, default='preprocessed_ref', help='folder containing motion_mean.npy and motion_std.npy')
    parser.add_argument('--silence_npy_path', type=str, default='reference_data/silence.npy', help='silence npy file path')
    parser.add_argument('--hierarchy_npy_path', type=str, default='reference_data/bvh_base.npy', help='hierarchy bvh base npy file path')

    ## Model setting 
    parser.add_argument('--mfcc_channel', type=int, default=26, help='wav mfcc channel')
    parser.add_argument('--context', type=int, default=10, help='context window')

    parser.add_argument('--hidden_size', type=int, default=256, help='LSTM hidden size')
    parser.add_argument('--dropout', type=float, default=0.1, help='LSTM dropout rate (Recommend 0.1-0.2)')
    parser.add_argument('--n_joint', type=int, default=71, help='num of motion keypoint')

    ## Train env setting 
    parser.add_argument('--batch_size', type=int, default=15, help='batch size for model')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate for model train')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay in optimizer')
    parser.add_argument('--milestones', nargs="+", type=int, default=[20, 25], help='weight decay in optimizer')
    parser.add_argument('--epochs', type=int, default=100, help='num of epochs train')
    parser.add_argument('--save_int', type=int, default=10, help='save ckpt interval')
    

    parser.add_argument('--revised_model', action="store_true", help='use the revised model version')

    return parser.parse_args()

def main():
    args = parse_args()
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('target device :', device)

    if not os.path.exists(args.results_path) : os.makedirs(args.results_path)
    save_path = get_save_path(args.results_path)
    print(f'model will save in {save_path}')

    with open(os.path.join(save_path, 'args.txt'), 'w') as args_file : 
        args_file.write(str(args))

    print(np.load("preprocessed_norm/MM_M_C_F_C_S064_001_Y.npy").max())

    dataset = None
    
    if (args.revised_model):
        print("Using REVISED Dataset (Normalized)")
        dataset = AudioGestureDatasetRevised(args.dataset_path, args.context, args.silence_npy_path, stats_path=args.stats_dir)
    else:
        print("Using ORIGINAL Dataset (Raw)")
        dataset = AudioGestureDataset(args.dataset_path, args.context, args.silence_npy_path)

    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, 
                            shuffle=True, collate_fn=Collator(args.silence_npy_path, args.hierarchy_npy_path).collate_fn)

    criterion = nn.L1Loss()
    # MSELoss()

    model = None

    # for revised model
    if (args.revised_model):
        print("Revised Model running")
        model = AudioGestureLSTMRevised(args.mfcc_channel, args.context, args.hidden_size, args.n_joint, args.silence_npy_path, device, dropout=args.dropout).to(device)
    else:
        print("Old Model running")
        model = AudioGestureLSTM(args.mfcc_channel, args.context, args.hidden_size, args.n_joint, args.silence_npy_path, device, dropout=args.dropout).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    scheduler = None

    if (args.revised_model):
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5,    # Cut LR in half
            patience=5,    # Wait only 5 epochs (instead of 10) before cutting
            threshold=0.1, # Improvement must be significant
            verbose=True
        )
    else:
        scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    writer = SummaryWriter(save_path)

    step = 0
    for epoch in range(args.epochs):
        # for schdeduler step
        running_loss = 0.0

        progress = tqdm.tqdm(dataloader)
        for wavs, xyzs in progress :
            model.train() 
            wavs = wavs.float().to(device)
            xyzs = xyzs.float().to(device)

            outputs = model(wavs)
            loss = criterion(outputs, xyzs)

            if step % 100 == 0: # Print only occasionally
                print(f"\n[Debug] Target Min: {xyzs.min():.3f}, Max: {xyzs.max():.3f}")
                print(f"[Debug] Output Min: {outputs.min():.3f}, Max: {outputs.max():.3f}")

            optimizer.zero_grad()
            loss.backward() 

            if (args.revised_model):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            progress.set_description(f'Epoch: {epoch+1:3d} / {args.epochs} | train loss: {loss.item():8.3f}')
            writer.add_scalar('train_loss', loss.item(), step)

            running_loss += loss.item()
            step += 1

        epoch_avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {epoch_avg_loss:.4f}")

        if (epoch + 1) % args.save_int == 0 : 
            target_file_path = os.path.join(save_path, f'LSTM_{epoch+1}.ckpt')
            torch.save(model.state_dict(), target_file_path)

        lr = optimizer.param_groups[-1]['lr']
        writer.add_scalar('lr', lr, epoch)

        if (args.revised_model):
            scheduler.step(epoch_avg_loss)
        else:
            scheduler.step()
    
    torch.save(model.state_dict(), os.path.join(save_path, f'LSTM_Final.ckpt'))

if __name__=='__main__':
    main()