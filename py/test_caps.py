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
    parser.add_argument("--batch_size", type = int, default = 20, help = "Batch size")
    parser.add_argument("--routing_iter", type = int, default = 5, help = "Dynamic routing iteration number")
    parser.add_argument("--save_time", type = int, default = 50, help = "Save generated img every <> batches")
    parser.add_argument("--gamma", type = float, default = 0.9996, help = "Exponential lr coefficient")
    parser.add_argument("--recons_ratio", type = float, default = 0.005, help = "The ratio of reconstruction error")
    parser.add_argument("-d", "--del_dir", action = "store_true", help = "Delete dir ./logs and start new tensorboard records")
    parser.add_argument("-c", "--cuda", default = False, action = "store_true", help = "Use CUDA to speed up training")
    parser.add_argument("-t", "--test", default = False, action = "store_true", help = "Run test with trained model")
    args = parser.parse_args()

    epoches     = args.epoches
    del_dir     = args.del_dir
    use_cuda    = args.cuda
    batch_size  = args.batch_size
    save_time   = args.save_time
    gamma       = args.gamma
    test        = args.test
    ratio       = args.recons_ratio

    cap = CapsNet(args.routing_iter, args.batch_size)
    recons = Recons()
    margin_loss_func = MarginLoss(0.999, 0.001)
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

    cap_opt = optim.Adam(cap.parameters(), lr = 3e-2)
    recon_opt = optim.Adam(recons.parameters(), lr = 1e-2)
    cap_sch = optim.lr_scheduler.ExponentialLR(cap_opt, gamma = gamma)
    recon_sch = optim.lr_scheduler.ExponentialLR(recon_opt, gamma = gamma)
    torch.autograd.set_detect_anomaly(True)
    acc_cnt = 0
    img_cnt = 0
    for i in range(epoches):
        for k, (bx, by) in enumerate(data_set):
            cap_opt.zero_grad()
            recon_opt.zero_grad()

            bx = Var(bx).cuda()
            by = by.cuda()
            y_cap = cap(bx)
            # ======= reconstruction loss ======
            recons_img = recons(y_cap, by)
            recons_loss = recons_loss_func(recons_img, bx.view(batch_size, -1))
            recons_loss.backward()
            recon_opt.step()
            recon_sch.step()

            # ======= margin loss ======
            bx = Var(bx).cuda()
            y_cap = cap(bx)
            margin_loss = margin_loss_func(y_cap, by) 
            recons_img = recons(y_cap, by)
            loss = margin_loss + ratio * recons_loss_func(recons_img, bx.view(batch_size, -1))
            loss.backward()
            cap_opt.step()
            cap_sch.step()

            train_cnt = i * batch_number + k
            img_cnt += batch_size
            acc_cnt += MarginLoss.accCounter(y_cap, by)
            acc = acc_cnt / img_cnt
            writer.add_scalar('Loss/Total Loss', loss, train_cnt)
            writer.add_scalar('Loss/Reconstruction loss', recons_loss, train_cnt)
            writer.add_scalar('Loss/Capsule loss', margin_loss, train_cnt)
            writer.add_scalar('Acc/Prediction Accuracy', acc, train_cnt)
            print("Batch %4d / %4d\t recons loss: %.4f\t total loss: %.4f\t acc: %.4f"%(
                k, batch_number, recons_loss.item(), loss.item(), acc
            ))

            if k % save_time == 0:
                img_to_save = recons_img.detach().view(batch_size, 1, 28, 28)
                save_image(img_to_save[:25], "..\\imgs\\G_%d.jpg"%(k + 1), nrow = 5, normalize = True)
    writer.close()
    print("Output completed.")
    