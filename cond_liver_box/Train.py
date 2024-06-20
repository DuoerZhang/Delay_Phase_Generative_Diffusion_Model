
import os
import time
import copy
import random
import warnings
import argparse
# import pydicom
import numpy as np
# import cv2
from PIL import Image
import PIL
import SimpleITK as sitk
from typing import Dict
from skimage.metrics import structural_similarity as ssim 
from skimage.metrics import peak_signal_noise_ratio as psnr

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10,ImageFolder
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F

import torchvision
from torch.optim.lr_scheduler import _LRScheduler, StepLR

from cond_liver_box.network import UNet
from cond_liver_box.CT_dataset import MPIDataset

from guided_diffusion.script_util import create_gaussian_diffusion
from guided_diffusion.nn import mean_flat,sum_flat
from Scheduler import GradualWarmupScheduler

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-9):
        self.power = power
        self.max_iters = max_iters # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # print(self.base_lrs, self.min_lr, ( 1 - self.last_epoch/self.max_iters )**self.power)
        return [ max( base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power, self.min_lr)
            for base_lr in self.base_lrs]

def loopy(dl):
    while True:
        for x in iter(dl): yield x
def mkdirs(path):
    if not os.path.exists(path):
        os.mkdir(path)
def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in target_dict.keys():
        if 'coords' in key:
            continue
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))

def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])

    # Data loading code
    train_dataset = MPIDataset(root = modelConfig["train_data_dir"])
    val_dataset = MPIDataset(root = modelConfig["val_data_dir"])
    print("Number of datassets", len(train_dataset),len(val_dataset))

    # train_sampler = None
    train_loader = loopy(DataLoader(train_dataset,
        batch_size = modelConfig["batch_size"],shuffle=True,
        num_workers=2,pin_memory=True,drop_last=True,
    ))
    val_loader = loopy(DataLoader(val_dataset,
        batch_size=2,shuffle=False,
        num_workers=1,pin_memory=True,drop_last=True,
    ))
    writer = SummaryWriter(log_dir=os.path.join(modelConfig["save_weight_dir"],"runs"))
    results_dir = modelConfig["save_weight_dir"] + '/results'
    mkdirs(results_dir)
    # # create model, model setup
    print("=> creating model")

    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    net_model.cuda(device)
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
   
    scheduler = PolyLR(optimizer, modelConfig["iters"], power=0.9)

    ema_model = copy.deepcopy(net_model)    
    # optionally resume from a checkpoint
    if modelConfig["training_load_weight"] is not None:
        if os.path.isfile(os.path.join(modelConfig["save_weight_dir"], modelConfig["training_load_weight"])):
            print("=> loading checkpoint '{}'".format(modelConfig["training_load_weight"]))
            checkpoint = torch.load(os.path.join(
                modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location='cpu')
            start_iters = checkpoint["iters"]
            net_model.load_state_dict(checkpoint["state_dict"], strict=False)
            ema_model.load_state_dict(checkpoint["ema_state_dict"], strict=False)
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (iterations {})".format(
                    modelConfig["training_load_weight"], checkpoint["iters"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(modelConfig["training_load_weight"]))
    else:
        start_iters = 0

    gaussian_diffusion = create_gaussian_diffusion(
        steps=modelConfig["T"],learn_sigma=True,
        noise_schedule='linear',use_kl=False,predict_xstart=False,rescale_timesteps=False,
        rescale_learned_sigmas=False,p2_gamma=1,p2_k=1,
    )

    eval_gaussian_diffusion = create_gaussian_diffusion(
        steps=modelConfig["T"],learn_sigma=True,
        noise_schedule='linear',use_kl=False,timestep_respacing="100",
        predict_xstart=False,rescale_timesteps=False,     
        rescale_learned_sigmas=False,p2_gamma=1,p2_k=1,
    )
    for iter in range(start_iters, modelConfig["iters"]):
        batch_time = AverageMeter("Time", ":6.3f")
        LearnR = AverageMeter("LR", ":.4e")
        losses = AverageMeter("Loss", ":.4e")
        mse = AverageMeter("MSE", ":.4e")
        progress = ProgressMeter(
            [batch_time, LearnR, losses, mse], prefix="Training: "
        )
        # measure data loading time
        end = time.time()
        batch = next(train_loader)
        img0, img1, img2, img3, name, mask,liver = batch
        img0 = img0.to(device, non_blocking=True)
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        img3 = img3.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        liver = liver.to(device, non_blocking=True)
        batch = img0.shape[0]
        
        diffusion_t = np.random.choice(modelConfig["T"], size=(batch,))
        diffusion_t = torch.from_numpy(diffusion_t).to(device).long()
        noise = torch.randn(size=[modelConfig["batch_size"], 1, modelConfig["img_size"], modelConfig["img_size"]], device=device)
        net_model.train()
        optimizer.zero_grad()

        loss = gaussian_diffusion.training_losses(net_model, img2, diffusion_t, model_kwargs={'x_art': img1, 'x_pv': img3,'mask':liver}, noise=noise)
        loss_mse = torch.mean(loss["mse"])

        loss_ = torch.mean(loss["loss"])
        loss_.backward()
        torch.nn.utils.clip_grad_norm_(net_model.parameters(), modelConfig["grad_clip"])
        optimizer.step()
        ema(net_model, ema_model, 0.999)
        
        # update measure info
        losses.update(loss_.item(), batch)
        mse.update(loss_mse.item(), batch)
        lr = optimizer.state_dict()['param_groups'][0]["lr"]
        LearnR.update(lr)
        batch_time.update(time.time() - end)
        end = time.time()
        scheduler.step()

        if iter % modelConfig["printfreq"] == 0:
            writer.add_scalars("train_loss", {'loss':loss_,'mse':loss_mse}, iter)
            writer.add_scalar("lr", lr, iter)
            progress.display(iter)
            
        if iter % modelConfig["testfreq"] == 0:
            print('-----eval in train-----')
            evalintrain(val_loader, eval_gaussian_diffusion, ema_model, iter, results_dir, modelConfig)
        if (iter+1) > 20000 and iter % modelConfig["savefreq"] == 0:
            print('-----saving inter model-----')
            torch.save({
                "iters": iter,
                "state_dict": net_model.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, os.path.join(modelConfig["save_weight_dir"], "checkpoint_%06d.pth.tar" % iter))
        if (iter) == modelConfig["iters"]:
            print('-----saving final model-----')
            torch.save({
                "iters": iter,
                "state_dict": net_model.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, os.path.join(modelConfig["save_weight_dir"], "checkpoint_%06d.pth.tar" % iter))
            print('-----Train finished!-----')

def evalintrain(val_loader, gaussian_diffusion, model, iter, results_dir, modelConfig):
    device = torch.device(modelConfig["device"])
    model.eval()

    batch = next(val_loader)
    img0, img1, img2, img3, name, mask,liver = batch
    img0 = img0.to(device, non_blocking=True)
    img1 = img1.to(device, non_blocking=True)
    img2 = img2.to(device, non_blocking=True)
    img3 = img3.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    liver = liver.to(device, non_blocking=True)

    batch = img0.shape[0]
    img = torch.randn((batch, 1, modelConfig["img_size"], modelConfig["img_size"]), device=device)

    indices = list(range(gaussian_diffusion.num_timesteps))[::-1]
    for i in indices:
        t = torch.tensor([i] * batch, device=device)
        with torch.no_grad():
            out = gaussian_diffusion.p_mean_variance(model, img, t, model_kwargs={'x_art': img1, 'x_pv': img3,'mask': liver})
            if i > 0:
                noise = torch.randn_like(img)
            else:
                noise = 0
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(img.shape) - 1)))
            )  # no noise when t == 0
            img = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            assert torch.isnan(img).int().sum() == 0, "nan in tensor."

    out_path_name = results_dir+ '/patch_'+str(iter)+'_'+str(name)+'.png'
    video_grid = make_grid(torch.cat([img1,img3,img2, img,mask]), nrow=2, normalize=True, value_range=(-1, 1),padding = 0)
    save_image(video_grid, out_path_name)
    

def eval(modelConfig):
    device = torch.device(modelConfig["device"])
      # Data loading code
    eval_dataset = MPIDataset(root=modelConfig["test_data_dir"])
    print("Number of eval datassets", len(eval_dataset))
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=None,
        drop_last=False,
    )
    # create model
    print("=> creating model")
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                    num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    print("=> creating model:Done!")
    net_model.cuda(device)
    ema_model = copy.deepcopy(net_model)
    
    load_path = os.path.join(modelConfig["save_weight_dir"], modelConfig["training_load_weight"])
    if os.path.isfile(load_path):
            print("=> loading checkpoint '{}'".format(modelConfig["training_load_weight"]))
            checkpoint = torch.load(load_path, map_location='cpu')
            net_model.load_state_dict(checkpoint["state_dict"], strict=False)
            ema_model.load_state_dict(checkpoint["ema_state_dict"], strict=False)
            # optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoches {})".format(
                    modelConfig["training_load_weight"], checkpoint["iters"]
                )
            )
    else:
        print("=> no checkpoint found at '{}'".format(modelConfig["training_load_weight"]))
    cudnn.benchmark = True
    
    eval_gaussian_diffusion = create_gaussian_diffusion(
        steps=modelConfig["T"],
        learn_sigma=True,
        noise_schedule='linear',
        use_kl=False,
        timestep_respacing="100",
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        p2_gamma=1,
        p2_k=1,
    )
    ema_model.eval()

    with tqdm(eval_loader, dynamic_ncols=True) as tqdmDataLoader:
        for batch in tqdmDataLoader:
            img0, img1, img2, img3, target, mask,liver = batch
            
            img0 = img0.to(device, non_blocking=True)
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            img3 = img3.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            liver = liver.to(device, non_blocking=True)
            batch = img0.shape[0]

            indices = list(range(eval_gaussian_diffusion.num_timesteps))[::-1]
            img = torch.randn((batch, 1, modelConfig["img_size"], modelConfig["img_size"]), device=device)

            for i in indices:
                t = torch.tensor([i] * batch, device=device)
                with torch.no_grad():
                    out = eval_gaussian_diffusion.p_mean_variance(ema_model, img, t, model_kwargs={'x_art': img1, 'x_pv': img3,'mask': liver})
                    if i > 0:
                        noise = torch.randn((batch, 1, modelConfig["img_size"], modelConfig["img_size"]), device=device)
                    else:
                        noise = 0
                    nonzero_mask = (
                        (t != 0).float().view(-1, *([1] * (len(img.shape) - 1)))
                    )  # no noise when t == 0
                    img = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            sampledName = 'Tres100_'
            video_grid = make_grid(torch.cat([img1,img3,img2,img,mask,liver]), nrow=6, normalize=True, value_range=(-1, 1),padding = 0)
            save_image(video_grid, os.path.join(
                modelConfig["sampled_dir"],  sampledName+str(target)+'.png'))

            out = img.cpu().numpy()
            out = sitk.GetImageFromArray(out.squeeze())
            sitk.WriteImage(out,os.path.join(modelConfig["sampled_dir"],sampledName+str(target)+'_pred.nii'))
            
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, meters, prefix=""):
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + "[{:07d}]".format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))