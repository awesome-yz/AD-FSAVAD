from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import ast
from torch.nn import functional as F
import os
import random
import torch.utils.data
import torchvision.utils as vutils
import torch.backends.cudnn as cudn
from torch.nn import functional as F
import imageio
from torch.nn import BCELoss, MSELoss, L1Loss
import ast
import argparse
from dataset import TestDataset
from utils import *
from models.discriminator import Discriminator
from models.generator_SimVP import UNet as Generator
import pytorch_msssim
import math
import psutil
from torch.nn import TripletMarginLoss as TML
from omegaconf import OmegaConf
import time

def wasserstein_loss(input):
    return torch.mean(input)

def create_mask(image):
    conv_2d = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=(1,1))
    out = conv_2d(image)
    mask = torch.argmax(out, dim=-1)
    return mask

def Load_Dataloader(train_path_list, tf, batch_size, device):
    data = TestDataset(train_path_list, device, tf)
    dataloader = DataLoader(data,batch_size=batch_size)
    return dataloader

def overall_generator_pass(generator, discriminator, img, gt, valid):
    recon_batch, out, noisy_imgs, noise = generator(img)
    recon_batch = recon_batch[0].unsqueeze(0) # [1, 3, 256, 256]

    msssim, f1, psnr = loss_function(recon_batch, gt)
    psnr = torchPSNR(recon_batch, gt)
    mse_loss = MSELoss()
    bce_loss = BCELoss()

    l2_pred = mse_loss(recon_batch, gt)
    l2_noise = mse_loss(noisy_imgs[1:], noise[1:])
    loss = msssim + f1 + l2_noise - (1.0* psnr)
    g_loss = bce_loss(discriminator(recon_batch), valid) + loss
    return g_loss, recon_batch, loss, msssim

def overall_discriminator_pass(discriminator, recon_batch, gt, valid, fake):
    bce_loss = BCELoss()
    real_loss = bce_loss(discriminator(gt), valid)
    fake_loss = bce_loss(discriminator(recon_batch.detach()), fake)
    d_loss =  (real_loss + fake_loss)/2
    return d_loss


def main(config):
    torch.manual_seed(1)
    # specify arguments
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Device: {device}')
    k_shots = config.k_shots
    print(f'Kshot: {k_shots}')
    num_tasks = config.num_tasks
    adam_betas = tuple(config.adam_betas)
    gen_lr = config.gen_lr
    model_folder_path = config.model_folder_path
    print(f'Model chkpts folder: {model_folder_path}')
    frame_path = config.frame_path
    print(f'Frames path: {frame_path}')
  
    batch_size = 1
    generator = Generator(batch_size, config['img_channels'], config['n_channels'],
                          config['g_channel_multipliers'], config['g_is_attention']) 
    discriminator = Discriminator(n_classes=config['d_n_classes'], resolution=config['d_resolution'])
    generator.to(device)
    discriminator.to(device)

    tf = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])

    generator.load_state_dict(torch.load(os.path.join(model_folder_path, "Generator_Final.pt"), map_location=device))
    discriminator.load_state_dict(torch.load(os.path.join(model_folder_path, "Discriminator_Final.pt"), map_location=device))

    optimizer_g = optim.Adam(generator.parameters(), lr=gen_lr, betas=adam_betas)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=gen_lr, betas=adam_betas)

    mse_loss = MSELoss()
    mae_loss = L1Loss()
    all_AUC = []
    avg_auc = 0

    test_path_list = createTestData(frame_path, k_shots, config.gt_folder, split=True)
    test_dataloader = Load_Dataloader(test_path_list, tf, batch_size, device)

    # creating graphs folder:
    if not os.path.exists(config['test_graphs_folder']):
        create_folder(config['test_graphs_folder'])

    total_score_mse = []
    total_score_mae = []
    total_score_mse_psnr = []
    total_score_mae_psnr = []
    _,_, video_list = test_path_list
    video=0
    video_test_times = []
    video_fine_tune_times = []

    train_start_time = time.perf_counter()
    for vid_frames, vid_labels in test_dataloader:
            norm_frames = []
            labels = []
            video += 1
            video_start_time = time.perf_counter()
            for k_idx, frame_sequence in enumerate(vid_frames[0]):
                print("video:k_shot {}:{} finetuning".format(video, k_idx+1))
                if len(frame_sequence[0])==3:
                    img = frame_sequence[0]
                    gt = frame_sequence[1]

                    img, gt, valid, fake = prep_data(img, gt, device)
                    norm_frames.append(gt)
                    labels.append(vid_labels[0][k_idx][0])

                    # Finetune generator
                    optimizer_g.zero_grad()
                    g_loss, recon_batch, loss, msssim = overall_generator_pass(generator, discriminator, img, gt, valid)
                    g_loss.backward()
                    optimizer_g.step()

                    # Finetune discriminator
                    optimizer_d.zero_grad()
                    d_loss = overall_discriminator_pass(discriminator, recon_batch, gt, valid, fake)
                    d_loss.backward()
                    optimizer_d.step()
            
            video_fine_tune_time = time.perf_counter() - video_start_time
            video_fine_tune_times.append(video_fine_tune_time)
            print("video: {} Finetuning time: ".format(video), video_fine_tune_time)
            ## Testing
            real_gt = []
            gt_label = []
            dist_mse_set = []
            dist_mae_set = []
            psnr_set = []
            

            with torch.no_grad():
                print("video: {} Testing".format(video))
                for t_idx, frame_sequence in enumerate(vid_frames[1]):
                
                    img = frame_sequence[0]
                    gt = frame_sequence[1]

                    img, gt, _, _, = prep_data(img, gt, device)
                    pred_gt, _, _, _ = generator(img)
                    pred_gt = pred_gt[0].unsqueeze(0)
                    
                    dist_mse = mse_loss(pred_gt, gt)
                    dist_mae = mae_loss(pred_gt, gt)
                    psnr_test = torchPSNR(pred_gt, gt)
                    

                    dist_mse_set.append(dist_mse.detach().cpu().item())
                    dist_mae_set.append(dist_mae.detach().cpu().item())
                    psnr_set.append(psnr_test.detach().cpu().item())
                    real_gt.append(gt)

                    # test_write_images(pred_gt, gt, config['test_images_folder'], t_idx, video)

                for t_idx, lbl in enumerate(vid_labels[1]):
                    label = lbl[0]
                    gt_label.append(label.item())

                norm_score_mse = normalize_score(dist_mse_set)
                norm_score_mae = normalize_score(dist_mae_set)
                norm_score_psnr = normalize_score_psnr(psnr_set)

                anomaly_score_mse = score_sum(norm_score_psnr,norm_score_mse, 0.8)
                anomaly_score_mae = score_sum(norm_score_psnr,norm_score_mae, 0.8)


                if len(gt_label)> len(anomaly_score_mse):
                    increase = len(gt_label) - len(anomaly_score_mse)
                    gt_label =gt_label[:-increase]
               
                if all(x == 1 for x in gt_label):
                    gt_label.append(0)
                    norm_score_mse.append(0)
                    norm_score_mae.append(0)
                    anomaly_score_mse.append(0)
                    anomaly_score_mae.append(0)

                # time for each video:
                print("Video {} testing time: ".format(video), time.perf_counter() - video_start_time)
                video_test_times.append(time.perf_counter() - video_start_time)
               
                # creating folder for each video:
                if not os.path.exists(os.path.join(config['test_graphs_folder'], str(video))):
                    create_folder(os.path.join(config['test_graphs_folder'], str(video)))
                
                # storing scores as np
                np.save(os.path.join(config['test_graphs_folder'], str(video), 'mae'), norm_score_mae)
                np.save(os.path.join(config['test_graphs_folder'], str(video), 'mse'), norm_score_mse)
                np.save(os.path.join(config['test_graphs_folder'], str(video), 'mae+psnr'), anomaly_score_mae)
                np.save(os.path.join(config['test_graphs_folder'], str(video), 'mse+psnr'), anomaly_score_mse)
                np.save(os.path.join(config['test_graphs_folder'], str(video), 'gt'), gt_label)
                
                vid_score_mse = AUC(norm_score_mse, gt_label)
                vid_score_mae = AUC(norm_score_mae, gt_label)
                vid_score_mse_psnr = AUC(anomaly_score_mse, gt_label)
                vid_score_mae_psnr = AUC(anomaly_score_mae, gt_label)
            
                print('Video {} AUC_MSE: {}'.format(video, vid_score_mse))
                print('Video {} AUC_MAE: {}'.format(video, vid_score_mae))
                print('Video {} AUC_MSE_PSNR: {}'.format(video, vid_score_mse_psnr))
                print('Video {} AUC_MAE_PSNR: {}'.format(video, vid_score_mae_psnr))

                total_score_mse.append(vid_score_mse)
                total_score_mae.append(vid_score_mae)
                total_score_mse_psnr.append(vid_score_mse_psnr)
                total_score_mae_psnr.append(vid_score_mae_psnr)

    print('Total testing time: ', time.perf_counter() - train_start_time)
    print('Average testing time per video: ', np.array(video_test_times).mean())
    print('Average finetuning time per video: ', np.array(video_fine_tune_times).mean())
   
    # average AUC across videos
    print('Avg AUC_MSE: {}'.format(np.array(total_score_mse).mean()))
    print('Avg AUC_MAE: {}'.format(np.array(total_score_mae).mean()))
    print('Avg AUC_MSE_PSNR: {}'.format(np.array(total_score_mse_psnr).mean()))
    print('Avg AUC_MAE_PSNR: {}'.format(np.array(total_score_mae_psnr).mean()))

                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="path to config")
    args = parser.parse_args()

    # config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    config = OmegaConf.load(args.config)
    main(config)