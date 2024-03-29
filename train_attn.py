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
from torch.nn import BCELoss, MSELoss, BCEWithLogitsLoss
import ast
import argparse
import yaml
from dataset import TrainDataset
from utils import createEpochData, loss_function, create_folder, prep_data, torchPSNR, write_images
from models.discriminator import Discriminator
from models.generator import Generator
import os
from matplotlib import pyplot as plt
from copy import deepcopy
import imageio
from torch.utils.tensorboard import SummaryWriter 
from omegaconf import OmegaConf

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('GroupNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.5)
        torch.nn.init.zeros_(m.bias)
    elif classname.find('ConvTranspose2d') !=-1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)


def Load_Dataloader(path_list, tf, batch_size, device, test=False):
    data = TrainDataset(path_list, tf, device)    
    dataloader = DataLoader(data,batch_size=batch_size)
    return dataloader


def overall_generator_pass(generator, discriminator, img, gt, real):
    # recon_out, x_t, noise = generator(img)
    recon_out = generator(img)
    # recon_out = recon_batch[0].unsqueeze(0) # [1, 3, 256, 256]
    msssim, f1, _ = loss_function(recon_out, gt)
    psnr_loss =  (-1.0) * torchPSNR(recon_out, gt)
    psnr = torchPSNR(recon_out, gt)
    mse_loss = MSELoss()
    bce_loss = BCELoss()

    # l2_noise= mse_loss(x_t, noise)
    # loss = msssim + f1 + l2_noise + psnr_loss
    # loss = f1 + l2_noise 
    # loss = msssim + f1 + psnr_loss
    loss = f1
    dis_out = discriminator(recon_out)

    g_loss = bce_loss(dis_out, real) 
    g_loss+= loss
    return g_loss, recon_out, loss, msssim, psnr

def overall_discriminator_pass(discriminator, recon_out, gt, real, fake):
    bce_loss = BCELoss()

    real_out = discriminator(gt)
    fake_out = discriminator(recon_out.detach())

    real_loss = bce_loss(real_out, real)
    fake_loss = bce_loss(fake_out, fake)
    d_loss =  (real_loss + fake_loss)/2
    return d_loss

def meta_update_model(model, optimizer, loss, gradients):
    # Register a hook on each parameter in the net that replaces the current dummy grad
    # with our grads accumulated across the meta-batch
    # GENERATOR
    hooks = []
    for (k,v) in model.named_parameters():
        def get_closure():
            key = k
            def replace_grad(grad):
                return gradients[key]
            return replace_grad
        hooks.append(v.register_hook(get_closure()))

    # Compute grads for current step, replace with summed gradients as defined by hook
    optimizer.zero_grad()
    loss.backward()

    # Update the net parameters with the accumulated gradient according to optimizer
    optimizer.step()

    # Remove the hooks before next training phase
    for h in hooks:
        h.remove()

"""MAIN TRAINING SCRIPT"""
def main(config):
    torch.manual_seed(1)
    # specify arguments
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Device: {device}')
    k_shots = config.k_shots
    num_tasks = config.num_tasks
    adam_betas = tuple(config.adam_betas)
    gen_lr = config.gen_lr
    total_epochs = config.total_epochs
    model_folder_path = config.model_folder_path
    train_frame_path = config.train_frame_path
    step_size=config.step_size
    images_folder = config.images_folder

    print(f'kshots: {k_shots}')
    print(f'Data path: {train_frame_path}')
    print(f'Model save path: {model_folder_path}')
    print(f'Training epochs: {total_epochs}')
    print(f'Output Images: {images_folder}')

    batch_size = 1
    # Initialize generator and discriminator
    generator = Generator(encoder_args=config.encoder_args,
                          decoder_args=config.decoder_args, 
                          diff_args=config.diff_args, 
                          sp_attn_args=config.sp_attn_args,
                          tp_attn_args = config.tp_attn_args,
                          ff_args= config.ff_args
                          )
    discriminator = Discriminator(**config.discriminator_args)
    generator.to(device=device)
    discriminator.to(device=device)

    # Tensorboard Folder
    create_folder(config.tensorboard_folder)
    writer = SummaryWriter(config.tensorboard_folder)
    print(f'Tensorboard directory: {config.tensorboard_folder}')

    # Training the Model
    # optimizer
    optimizer_G = optim.Adam (generator.parameters(), lr= gen_lr,  betas=adam_betas)
    optimizer_D = discriminator.optim 
    scheduler_G = lr_scheduler.StepLR(optimizer_G, step_size=step_size)
    scheduler_D = lr_scheduler.StepLR(optimizer_D, step_size=step_size)

    tf = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
    create_folder(model_folder_path)  
    generator_path = os.path.join(model_folder_path, str.format("Generator_previous.pt"))
    discriminator_path = os.path.join(model_folder_path, str.format("Discriminator_previous.pt"))

    torch.save(generator.state_dict(), generator_path)
    torch.save(discriminator.state_dict(), discriminator_path)
    previous_generator = generator_path
    previous_discriminator = discriminator_path

    # Set Up Training Loop
    g_tr_loss = []
    g_vl_loss = []
    d_tr_loss = []
    d_vl_loss = []


    import pdb; pdb.set_trace()
    for epoch in range(total_epochs):
        train_path_list = createEpochData(train_frame_path, num_tasks, k_shots)
        train_dataloader = Load_Dataloader(train_path_list, tf, batch_size, device=device)

        for _, epoch_of_tasks in enumerate(train_dataloader):
            gen_epoch_grads = []
            dis_epoch_grads = []

            print("Epoch: ", epoch+1)
            # ------------------------------Meta-Training--------------------
            for tidx, task in enumerate(epoch_of_tasks):
                print ('\n Meta Training \n')
                generator.load_state_dict(torch.load(previous_generator))
                discriminator.load_state_dict(torch.load(previous_discriminator))

                inner_optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=adam_betas)
                inner_optimizer_D = discriminator.optim
                inner_scheduler_G = lr_scheduler.StepLR(inner_optimizer_G, step_size=step_size)
                inner_scheduler_D = lr_scheduler.StepLR(inner_optimizer_D, step_size=step_size)

                print("Task: ", tidx+1)
                for kidx, frame_sequence in enumerate(task[:k_shots]):
                    print('k-Shot Training: ', kidx)
                    # Configure input
                    img = frame_sequence[0]
                    gt = frame_sequence[1]

                    img, gt, valid, fake = prep_data(img, gt, device)

                    # Train Generator
                    inner_optimizer_G.zero_grad()
                    g_loss, recon_batch, loss, msssim, psnr = overall_generator_pass(generator, discriminator, img, gt, valid)
                    g_loss.backward()
                    inner_optimizer_G.step()
                    g_tr_loss.append(g_loss.item())

                    # Save generated images
                    create_folder(config['images_folder'])
                    write_images(recon_batch, gt, config['images_folder'], tidx+1, epoch)

                    # Train Discriminator
                    inner_optimizer_D.zero_grad()
                    d_loss = overall_discriminator_pass(discriminator, recon_batch, gt, valid, fake)
                    d_loss.backward()
                    inner_optimizer_D.step()
                    d_tr_loss.append(d_loss.item())

                    print ('Training: Epoch [{}/{}], Task [{}/{}], Reconstruction_Loss: {:.4f}, G_Loss: {:.4f}, D_loss: {:.4f}, msssim:{:.4f}, psnr: {:.4f} '
                           .format(epoch+1, total_epochs, tidx+1, num_tasks, loss.item(), g_loss, d_loss, msssim, psnr))
                    
                #-------------------Meta-Validation -----------------------
                print ('\n Meta Validation \n')
                # Store Loss Values
                gen_validation_loss_store = 0.0
                dis_validation_loss_store = 0.0
                gen_validation_loss = 0.0
                dis_validation_loss = 0.0
                
                dummy_frame_sequence = []
                # forward pass
                for vidx, val_frame_sequence in enumerate(task[-k_shots:]):
                    print(vidx)
                    if vidx == 0:
                        dummy_frame_sequence = val_frame_sequence
                    
                    img = val_frame_sequence[0]
                    gt = val_frame_sequence[1]
                    img, gt, valid, fake = prep_data(img, gt, device)
               
                    # k-Validation Generator
                    inner_optimizer_G.zero_grad()
                    g_loss, recon_batch, loss, msssim, psnr = overall_generator_pass(generator, discriminator, img, gt, valid)
                    g_vl_loss.append(g_loss.item())
                    
                    # k-Validation Discriminator
                    inner_optimizer_D.zero_grad()
                    d_loss = overall_discriminator_pass(discriminator, recon_batch, gt, valid, fake)
                    d_vl_loss.append(d_loss.item())
                  
                    # Store Loss Items to reduce memory usage
                    gen_validation_loss_store += g_loss.item()
                    dis_validation_loss_store += d_loss.item()

                    if (vidx == k_shots-1):
                        # Store the loss
                        gen_validation_loss = g_loss
                        dis_validation_loss = d_loss
                        gen_validation_loss.data = torch.FloatTensor([gen_validation_loss_store/k_shots]).to(device=device)
                        dis_validation_loss.data = torch.FloatTensor([dis_validation_loss_store/k_shots]).to(device=device)
                    
                    print("Generator Validation Loss: ", g_loss.item())
                    print("Discriminator Validation Loss: ", d_loss.item())

                # Compute Validation Grad
                print("Memory Allocated: ",torch.cuda.memory_allocated()/1e9)

                gen_grads = torch.autograd.grad(gen_validation_loss, generator.parameters())
                dis_grads = torch.autograd.grad(dis_validation_loss, discriminator.parameters())
                
                gen_meta_grads = {name:g for ((name, _), g) in zip(generator.named_parameters(), gen_grads)}
                dis_meta_grads = {name:g for ((name, _), g) in zip(discriminator.named_parameters(), dis_grads)}
                
                gen_epoch_grads.append(gen_meta_grads)
                dis_epoch_grads.append(dis_meta_grads)

            # inner_scheduler step
            inner_scheduler_G.step()
            inner_scheduler_D.step()

            # Meta Update
            print('\n Meta update \n')

            generator.load_state_dict(torch.load(previous_generator))
            discriminator.load_state_dict(torch.load(previous_discriminator))
            
            # Configure input
            img = dummy_frame_sequence[0]
            gt = dummy_frame_sequence[1]
            img, gt, valid, fake = prep_data(img, gt, device)

            # Dummy Forward Pass
            g_loss, recon_batch, loss, msssim, psnr = overall_generator_pass(generator, discriminator, img, gt, valid)
            d_loss = overall_discriminator_pass(discriminator, recon_batch, gt, valid, fake)

            # Unpack the list of grad dicts
            gen_gradients = {k: sum(d[k] for d in gen_epoch_grads) for k in gen_epoch_grads[0].keys()}
            dis_gradients = {k: sum(d[k] for d in dis_epoch_grads) for k in dis_epoch_grads[0].keys()}
            
            meta_update_model(generator, optimizer_G, g_loss, gen_gradients)
            meta_update_model(discriminator, optimizer_D, d_loss, dis_gradients)

            scheduler_G.step()
            scheduler_D.step()

            # Save the Model
            torch.save(generator.state_dict(), previous_generator)
            torch.save(discriminator.state_dict(), previous_discriminator)
            if (epoch % 50 == 0):
                writer.add_scalar('Generator Training Loss', torch.FloatTensor(g_tr_loss).sum()/ (k_shots * num_tasks * epoch), epoch)
                writer.add_scalar('Discriminator Training Loss', torch.FloatTensor(d_tr_loss).sum()/ (k_shots * num_tasks * epoch), epoch)
                writer.add_scalar('Generator Validation Loss', torch.FloatTensor(g_vl_loss).sum()/ (k_shots * num_tasks * epoch), epoch)
                writer.add_scalar('Discriminator Validation Loss', torch.FloatTensor(d_vl_loss).sum()/ (k_shots * num_tasks * epoch), epoch)
            if (epoch % 500 == 0):
                gen_path = os.path.join(model_folder_path, str.format("Generator_{}.pt", epoch+1))
                dis_path = os.path.join(model_folder_path, str.format("Discriminator_{}.pt", epoch+1))
                torch.save(generator.state_dict(), gen_path)
                torch.save(discriminator.state_dict(), dis_path)


    print("Training Complete")
    # Save final model 
    gen_path = os.path.join(model_folder_path, str.format("Generator_Final.pt"))
    dis_path = os.path.join(model_folder_path, str.format("Discriminator_Final.pt"))
    torch.save(generator.state_dict(), gen_path)
    torch.save(discriminator.state_dict(), dis_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to configuration")
    args = parser.parse_args()

    # config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    config = OmegaConf.load(args.config)
   
    main(config)
