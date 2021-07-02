#-*-coding:utf-8-*-
"""
    MNIST Caps
    @date 2021.6.30
    @author Qianyue He
"""
import os
import torch
import shutil
import argparse
from datetime import datetime
from torch.autograd import Variable as Var
from torchvision.utils import save_image
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
from torchvision import transforms

from torch import optim
from torch import nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from capsnet import CapsNet
from margin_loss import MarginLoss
from reconstruct import Recons

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoches", type = int, default = 40, help = "Training lasts for . epoches")
    parser.add_argument("--max_iter", type = int, default = 20, help = "max iteration number")
    parser.add_argument("--batch_size", type = int, default = 20, help = "Batch size")
    parser.add_argument("--routing_iter", type = int, default = 5, help = "Dynamic routing iteration number")
    parser.add_argument("--save_time", type = int, default = 200, help = "Save generated img every <> batches")
    parser.add_argument("--gamma", type = float, default = 0.95, help = "Exponential lr coefficient")
    parser.add_argument("-d", "--del_dir", action = "store_true", help = "Delete dir ./logs and start new tensorboard records")
    parser.add_argument("-c", "--cuda", default = False, action = "store_true", help = "Use CUDA to speed up training")
    parser.add_argument("-t", "--test", default = False, action = "store_true", help = "Run test with trained model")
    args = parser.parse_args()

    epoches     = args.epoches
    del_dir     = args.del_dir
    use_cuda    = args.cuda
    max_iter    = args.max_iter
    batch_size  = args.batch_size
    save_time   = args.save_time
    gamma       = args.gamma
    test        = args.test
    cap = CapsNet(args.routing_iter, args.batch_size)
    recons = Recons()
    margin_loss_func = MarginLoss()
    recons_loss_func = nn.MSELoss()

    tf = transforms.ToTensor()
    data_set = DataLoader(
        datasets.MNIST(
            "..\\..\\TorchLearning\\data\\",
            train = (not test),
            download = False,
            transform = tf
        ),
        batch_size = batch_size,
        shuffle = True,
    )

    if use_cuda and torch.cuda.is_available():
        print("Using CUDA.")
        cap = cap.cuda()
        cap.setCuda()
        recons = recons.cuda()
        margin_loss_func = margin_loss_func.cuda()
        recons_loss_func = recons_loss_func.cuda()
    else:
        print("CUDA not available.")
        cap = cap.cpu()
        recons = recons.cpu()
        margin_loss_func = margin_loss_func.cpu()
        recons_loss_func = recons_loss_func.cpu()
    
    logdir = '../logs/'
    if os.path.exists(logdir) and del_dir:
        shutil.rmtree(logdir)
    time_stamp = "{0:%Y-%m-%d/%H-%M-%S}-epoch{1}/".format(datetime.now(), epoches)
    writer = SummaryWriter(log_dir = logdir+time_stamp)
    batch_number = data_set.__len__()

    cap_opt = optim.Adam(cap.parameters(), lr = 1e-3)
    recon_opt = optim.Adam(recons.parameters(), lr = 1e-3)
    cap_sch = optim.lr_scheduler.ExponentialLR(cap_opt, gamma = gamma)
    recon_sch = optim.lr_scheduler.ExponentialLR(recon_opt, gamma = gamma)

    for i in range(epoches):
        for k, (bx, by) in enumerate(data_set):
            bx = Var(bx).cuda()
            by = by.cuda()
            y_cap = cap(bx)
            recons_img = recons(y_cap, by)
            print(by)
            print("By shape:", by.shape, ", recons shape:", recons_img.shape)

            recons_loss = recons_loss_func(recons_img, bx.view(batch_size, -1))
            recon_opt.zero_grad()
            recons_loss.backward()
            recon_sch.step(i)

            cap_opt.zero_grad()
            margin_loss = margin_loss_func(y_cap, by) 
            loss = margin_loss + recons_loss_func(recons_img, bx.view(batch_size, -1))
            loss.backward()
            cap_opt.step(i)

            train_cnt = i * batch_number + k
            writer.add_scalar('Loss/Total Loss', loss, train_cnt)
            writer.add_scalar('Loss/Reconstruction loss', recons_loss, train_cnt)
            writer.add_scalar('Loss/Capsule loss', margin_loss, train_cnt)
            print("Batch %4d / %4d\t recons loss: %.3f\t total loss: %.3f"%(
                k, batch_number, recons_loss.item(), loss.item()
            ))

            if k % save_time == 0:
                img_to_save = recons_img.detach().clamp_(0, 1).view((-1, 28, 28))
                save_image(img_to_save, "..\\imgs\\G_%d.jpg"%(i + 1), 1)
    writer.close()
    print("Output completed.")
    