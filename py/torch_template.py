#-*-coding:utf-8-*-
"""
    This is a template for torch main.py
"""
import os
import torch
import shutil
import argparse
from datetime import datetime
from torch.autograd import Variable as Var
from torchvision.utils import save_image

from torch import optim
from torch import nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoches", type = int, default = 40, help = "Training lasts for . epoches")
    parser.add_argument("--max_iter", type = int, default = 20, help = "max iteration number")
    parser.add_argument("-d", "--del_dir", action = "store_true", help = "Delete dir ./logs and start new tensorboard records")
    parser.add_argument("-c", "--cuda", default = False, action = "store_true", help = "Use CUDA to speed up training")
    args = parser.parse_args()

    epoches = args.epoches
    del_dir = args.del_dir
    use_cuda = args.cuda
    max_iter = args.max_iter

    tf = transforms.ToTensor()

    if use_cuda and torch.cuda.is_available():
        pass
    else:
        print("CUDA not available.")
    
    logdir = '../logs/'
    if os.path.exists(logdir) and del_dir:
        shutil.rmtree(logdir)
    time_stamp = "{0:%Y-%m-%d/%H-%M-%S}-epoch{1}/".format(datetime.now(), epoches)
    writer = SummaryWriter(log_dir = logdir+time_stamp)

    for i in range(epoches):
        pass
        # save_image(gen.detach().clamp_(0, 1), "..\\imgs\\G_%d.jpg"%(epoch + 1), 1)
    writer.close()
    print("Output completed.")
    