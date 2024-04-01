import os
import random
import torch
import pytorch_msssim
import numpy as np
from math import log10
import math
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torch.cuda import FloatTensor as Tensor
from torch.autograd import Variable
from collections import OrderedDict
import imageio
import torch.nn as nn
from torch import einsum
from einops import rearrange
import cv2 as cv2
import torch.backends.cudnn as cudnn
import logging
from PIL import Image
import cv2 as cv
from einops.layers.torch import Rearrange

# 345678
random_gen = np.random.default_rng(5678654)


os.environ['CUDA_LAUNCH_BLOCKING']= "1"
def generate_k_shot_frames(video_folder, k_shots):
    all_frames = sorted([x for x in os.listdir(video_folder) if x.endswith('.jpeg')])

    video_name = video_folder[-7:]      
    video_length = len(all_frames)

    #range(3, video_length)
    frame_samples = random.sample(range(3,video_length), k_shots * 2) # first k are meta-training, rest are meta-testing

    # range(0,4)
    k_frame_sequences = [[all_frames[v_index - before] for before in reversed(range(0,4))] for v_index in frame_samples]

    return video_folder, k_frame_sequences
    
def generate_test_frames(video_folder, k_shots, gt_folder, split=False):
    """
    Generating frames for testing
    """
    all_frames = sorted(os.listdir(video_folder))
    video_name = video_folder[-7:]
    all_gt = np.load(os.path.join(gt_folder, video_name+".npy"))

    num_frames = len(all_frames)
    frame_samples = [all_frames[i:i + 4] for i in range(0, num_frames, 4)]
    gt_sequences = [all_gt[i:i+4] for i in range(0, num_frames, 4)]


    if split:
        # get all normal frames
        indx = sorted([i for i,x in enumerate(frame_samples) if (gt_sequences[i]==0).all()])
        # randomize samples
        # np.random.seed(43)
        indx = random_gen.permutation(indx)
        k_shot_ind = indx[:k_shots]
        q_ind = sorted([i for i,x in enumerate(frame_samples) if not i in k_shot_ind])

        k_shot_frames = [frame_samples[i] for i in k_shot_ind]
        k_shot_gt = [gt_sequences[i] for i in k_shot_ind]

        frame_samples = [frame_samples[i] for i in q_ind]
        gt_sequences = [gt_sequences[i] for i in q_ind]


        return video_folder, k_shot_frames, k_shot_gt, frame_samples, gt_sequences
    else:
        return video_folder, frame_samples, gt_sequences
    

def generate_test_frames_ped(video_folder, k_shots, gt_folder, split=False):
    """
    Generating frames for testing
    """
    all_frames = sorted([x for x in os.listdir(video_folder) if x.endswith('.jpg')])
    video_name = video_folder[-2:]
    all_gt = np.load(os.path.join(gt_folder, video_name+".npy"))

    num_frames = len(all_frames)
    frame_samples = [all_frames[i:i + 4] for i in range(0, num_frames, 4)]
    gt_sequences = [all_gt[i:i+4] for i in range(0, num_frames, 4)]

    if split:
        # get all normal frames
        indx = sorted([i for i,x in enumerate(frame_samples) if (gt_sequences[i]==0).all()])
        # randomize samples
        # np.random.seed(85)
        indx = random_gen.permutation(indx)

        k_shot_ind = indx[:k_shots]
        q_ind = sorted([i for i,x in enumerate(frame_samples) if not i in k_shot_ind])

        k_shot_frames = [frame_samples[i] for i in k_shot_ind]
        k_shot_gt = [gt_sequences[i] for i in k_shot_ind]

        frame_samples = [frame_samples[i] for i in q_ind]
        gt_sequences = [gt_sequences[i] for i in q_ind]

        
        return video_folder, k_shot_frames, k_shot_gt, frame_samples, gt_sequences
    else:
        return video_folder, frame_samples, gt_sequences
    

def createEpochData(frame_path, numTasks, k_shots):
    dirs = sorted(os.listdir(frame_path))
  
    # Selected Tasks (videos that are being used)
    samples = random.sample(dirs, numTasks)
    selected_videos = [os.path.join(frame_path, x) for x in samples]

    train_path_list = []
    # task_order = [0, 2, 3, 4, 7, 12]
    train_curr_paths = []
    # for task in range(len(task_order)):
    for task in range(numTasks):
        video = selected_videos[task]
        video_folder, k_shot_frames = generate_k_shot_frames(video, k_shots)
        train_curr_paths.append([[os.path.join(frame_path, str(video_folder), ind) for ind in frame] for frame in k_shot_frames])
        
    train_path_list.append(train_curr_paths)
    return train_path_list

    
def createTestData(frame_path, k_shots, gt_folder=None, split=False):
    videos = sorted(os.listdir(frame_path))

    all_frame_paths = []
    all_gt_paths = []
    all_videos_names = []
    for vid in videos:
        k_shot_paths = []
        k_gt = []
        q_paths = []
        q_gt = []
        if split: 
            video_folder, k_shot_frames, k_shot_gt, frame_sequences, gt_sequences = generate_test_frames(os.path.join(frame_path, vid), k_shots, gt_folder, split=True)

            k_shot_paths.append([[os.path.join(frame_path, str(video_folder), ind) for ind in frame] for frame in k_shot_frames])
            q_paths.append([[os.path.join(frame_path, str(video_folder), ind) for ind in frame] for frame in frame_sequences])
            k_shot_paths += q_paths 

            k_gt.append(k_shot_gt)
            q_gt.append(gt_sequences)
            k_gt += q_gt
    
            all_frame_paths.append(k_shot_paths)
            all_gt_paths.append(k_gt)
            all_videos_names.append(video_folder)
        else:
            # needs to be reviewed
            video_folder, frame_sequences, gt_sequences = generate_test_frames(os.path.join(frame_path, vid), k_shots, gt_folder)

            q_paths.append([[os.path.join(frame_path, str(video_folder), ind) for ind in frame] for frame in frame_sequences])

            q_gt.append(gt_sequences)

            all_frame_paths.append(q_paths)
            all_gt_paths.append(q_gt)
       
    return all_frame_paths, all_gt_paths, all_videos_names

def get_norm_frame(frame_path, name, transform):
    videos = os.listdir(frame_path)
    vid_list = [x for x in videos if x.startswith(name)]
    video_folder = os.path.join(frame_path, random.sample(vid_list,1)[0])
    frames = os.listdir(video_folder) 
    sample = random.sample(frames, 1)[0]
    path = os.path.join(video_folder, sample)
    im_opened = Image.open(path).convert('RGB')
    im_tf = transform(im_opened).to(device='cuda')
    return im_tf


def createTestDataPed(frame_path, k_shots, gt_folder=None, split=False):
    videos = sorted(os.listdir(frame_path))

    all_frame_paths = []
    all_gt_paths = []
    all_videos_names = []
    for vid in videos:
        k_shot_paths = []
        k_gt = []
        q_paths = []
        q_gt = []
        if split: 
            video_folder, k_shot_frames, k_shot_gt, frame_sequences, gt_sequences = generate_test_frames_ped(os.path.join(frame_path, vid), k_shots, gt_folder, split=True)

            k_shot_paths.append([[os.path.join(frame_path, str(video_folder), ind) for ind in frame] for frame in k_shot_frames])
            q_paths.append([[os.path.join(frame_path, str(video_folder), ind) for ind in frame] for frame in frame_sequences])
            k_shot_paths += q_paths 

            k_gt.append(k_shot_gt)
            q_gt.append(gt_sequences)
            k_gt += q_gt
    
            all_frame_paths.append(k_shot_paths)
            all_gt_paths.append(k_gt)
            all_videos_names.append(video_folder)
        else:
            # needs to be reviewed
            video_folder, frame_sequences, gt_sequences = generate_test_frames_ped(os.path.join(frame_path, vid), k_shots, gt_folder)

            q_paths.append([[os.path.join(frame_path, str(video_folder), ind) for ind in frame] for frame in frame_sequences])

            q_gt.append(gt_sequences)

            all_frame_paths.append(q_paths)
            all_gt_paths.append(q_gt)
       
    return all_frame_paths, all_gt_paths, all_videos_names

def torchPSNR(recon_x, x):
    imdff = torch.clamp(recon_x, 0, 1) - torch.clamp(x, 0, 1)
    rmse = (imdff**2).mean().sqrt()
    ps = 10*torch.log10(1/rmse)
    return ps

def loss_function(recon_x, x):
    msssim = (1-pytorch_msssim.msssim(x,recon_x))/2
    f1 =  F.l1_loss(recon_x, x)
    # psnr_error=(10 * log10( 65025/ ((torch.abs(torch.sum(x) - torch.sum(recon_batch))))))
    psnr_error=(10 * log10( 65025/ ((torch.abs(torch.sum(x) - torch.sum(recon_x))))))

    return msssim, f1, psnr_error

def roll_axis(img):
    img = np.rollaxis(img, -1, 0)
    img = np.rollaxis(img, -1, 0)
    return img

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False

def prep_data(img, gt, device, gen_labels=True):
    if gen_labels:
        # Adversarial ground truths
        valid = torch.full((1,1), 0.9, requires_grad=False, dtype=torch.float32, device=device)
        fake = torch.full((1,1), 0.1, requires_grad=False, dtype=torch.float32, device=device)
    img = [Variable(img[x].to(device=device)) for x in range(len(img))]
    gt_tensor = Variable(gt[0]).to(device=device)
    return img, gt_tensor, valid, fake


def network_parameters(nets):
    num_params = sum(param.numel() for param in nets.parameters())
    return num_params

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if not k.startswith("swin_unet.prelu"):  # removing prelu weight and bias
                name = k[10:]  # remove `module.`
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

def img_conversion(input):
    imgs = input.data.cpu().numpy()[0, :] 
    imgs = roll_axis(imgs)
    input = (imgs * 255).astype(np.uint8)
    return input


def write_images(input, groundtruth, folder, task, epoch):
    # writing reconstructed input
    print('saving images for epoch: {}'.format(epoch))
    input = img_conversion(input)
    groundtruth = img_conversion(groundtruth)
    # imageio.imwrite(os.path.join(folder, 'task{}_epoch{}_pred.png'.format(task, epoch)), input)
    # imageio.imwrite(os.path.join(folder, 'task{}_epoch{}_gt.png'.format(task, epoch)), groundtruth)
    imageio.imwrite(os.path.join(folder, 'task{}_pred.png'.format(task)), input)
    imageio.imwrite(os.path.join(folder, 'task{}_gt.png'.format(task)), groundtruth)
    


def test_write_images(input, groundtruth, folder, frame, vid):

    if not os.path.exists(os.path.join(folder, str(vid))):
        os.makedirs(os.path.join(folder, str(vid)))
    # writing reconstructed input
    print('saving images for video: {}'.format(vid))
    input = img_conversion(input)
    groundtruth = img_conversion(groundtruth)

    gray_gt = cv.cvtColor(groundtruth, cv.COLOR_BGR2GRAY)
    gray_input = cv.cvtColor(input, cv.COLOR_BGR2GRAY)

    sub_img = cv.subtract(gray_gt, gray_input)

    thresh_img = cv.threshold(sub_img, 150, 255,
    cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    blur_img = cv.GaussianBlur(thresh_img, (13,13), 11)
    heatmap_pred = cv.applyColorMap(blur_img, cv.COLORMAP_JET)
    super_imposed_img = cv.addWeighted(heatmap_pred, 0.3, groundtruth, 0.7, 0)


    imageio.imwrite(os.path.join(folder,str(vid), 'image{}_pred.png'.format(frame)), input)
    imageio.imwrite(os.path.join(folder,str(vid), 'image{}_gt.png'.format(frame)), groundtruth)
    imageio.imwrite(os.path.join(folder, str(vid),  'image{}_heatmap.png'.format(frame)), super_imposed_img)
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

def print_log(message):
    print(message)
    logging.info(message)

def output_namespace(namespace):
    configs = namespace.__dict__
    message = ''
    for k, v in configs.items():
        message += '\n' + k + ': \t' + str(v) + '\t'
    return message

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)




def psnr(mse):
    # mse = mse.detach().cpu().numpy()
    psnr = 10 * math.log10(1 / mse)
    psnr = torch.tensor(psnr).cuda()

    return psnr

def acc(preds,gts):
    loss_func_mse = nn.MSELoss(reduction='none')
    psnr_list = []

    for idx in range(len(preds)):
        pred,gt= preds[idx],gts[idx]
        mse_imgs = torch.mean(loss_func_mse((pred +1)/2, (gt +1)/2))
        psnr_list.append(psnr(mse_imgs))

    return psnr_list

def anomaly_score(psnr, max_psnr, min_psnr):

    return ((psnr - min_psnr) / (max_psnr-min_psnr))

def anomaly_score_inv(psnr, max_psnr, min_psnr):
    # return (1.0 - ((psnr - min_psnr) / (max_psnr-min_psnr)))
    return ((psnr - min_psnr) / (max_psnr-min_psnr))

def anomaly_score_list_inv(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))

    return anomaly_score_list


def anomaly_score_list(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))

    return anomaly_score_list

def score_sum(psnr_set, dist_set, alpha):
    list_result = []
    for i in range(len(psnr_set)):
        list_result.append((alpha * dist_set[i] + (1 - alpha) * psnr_set[i]))
        # list_result.append((dist_set[i] + alpha * psnr_set[i]))

    # return list1,list2,list_result
    return list_result

def normalize_score(score_list):
    anomaly_score_list = list()
    list_min = np.min(score_list)
    list_max = np.max(score_list)

    for i in range(len(score_list)):
        anomaly_score_list.append(anomaly_score(score_list[i], list_max, list_min))
    return anomaly_score_list

def normalize_score_psnr(score_list):
    anomaly_score_list = list()
    list_min = np.min(score_list)
    list_max = np.max(score_list)

    for i in range(len(score_list)):
        anomaly_score_list.append(1.0 - anomaly_score(score_list[i], list_max, list_min))
    return anomaly_score_list


def AUC(anomal_scores, labels):
    frame_auc = roc_auc_score(y_true=labels, y_score=anomal_scores)
    return frame_auc

def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
    

def patch_embedding(patch_height, patch_width, dim, channels=3):
    patch_dim = channels * patch_height * patch_width
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
        nn.LayerNorm(patch_dim), # normalization 
        nn.Linear(patch_dim, dim), # projection to dim
        nn.LayerNorm(dim)
    )

def get_params_true(model):
    import pdb; pdb.set_trace()
    params_list = nn.ParameterList()
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_list.append(param)
    return params_list
