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
from confusion import ConfusionMatrix

from capsnet import CapsNet
from margin_loss import MarginLoss

if __name__ == "__main__":
    path = "../models/capsnet.pth"
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default = 5, help = "Training lasts for . epochs")
    parser.add_argument("--batch_size", type = int, default = 20, help = "Batch size")
    parser.add_argument("--routing_iter", type = int, default = 3, help = "Dynamic routing iteration number")
    parser.add_argument("--save_time", type = int, default = 50, help = "Save generated img every <> batches")
    parser.add_argument("--gamma", type = float, default = 1.0, help = "Exponential lr coefficient")
    parser.add_argument("--recons_ratio", type = float, default = 0.0005, help = "The ratio of reconstruction error")
    parser.add_argument("--test_time", type = int, default = 50, help = "Test frequency for validation set")
    parser.add_argument("-d", "--del_dir", action = "store_true", help = "Delete dir ./logs and start new tensorboard records")
    parser.add_argument("-c", "--cuda", default = False, action = "store_true", help = "Use CUDA to speed up training")
    parser.add_argument("-l", "--load", default = False, action = "store_true", help = "Load the trained '.pth' model")
    args = parser.parse_args()

    epochs     = args.epochs
    del_dir     = args.del_dir
    use_cuda    = args.cuda
    batch_size  = args.batch_size
    save_time   = args.save_time
    gamma       = args.gamma
    ratio       = args.recons_ratio
    test_time   = args.test_time
    load_model  = args.load

    cap = CapsNet(args.routing_iter, args.batch_size)
    margin_loss_func = MarginLoss(0.9, 0.1)
    recons_loss_func = nn.MSELoss(size_average = False)

    tf = transforms.ToTensor()
    data_set = DataLoader(
        datasets.MNIST("..\\..\\TorchLearning\\data\\", 
            train = True, download = False, transform = tf),
        batch_size = batch_size, shuffle = True,
    )

    test_set = DataLoader(
        datasets.MNIST("..\\..\\TorchLearning\\data\\", 
            train = False, download = False, transform = tf),
        batch_size = batch_size, shuffle = True,
    )

    if use_cuda and torch.cuda.is_available():
        print("Using CUDA.")
        cap = cap.cuda()
        cap.setCuda()
        margin_loss_func = margin_loss_func.cuda()
        recons_loss_func = recons_loss_func.cuda()
    else:
        print("CUDA not available.")
        cap = cap.cpu()
        margin_loss_func = margin_loss_func.cpu()
        recons_loss_func = recons_loss_func.cpu()
    
    if load_model == True:
        if os.path.exists(path) == False:
            print("No such file as '%s'! Model failed to load."%(path))
        else:
            save = torch.load(path)   # 保存的优化器以及模型参数
            save_model = save['model']                  # 保存的模型参数
            model_dict = cap.state_dict()              # 当前网络参数
            state_dict = {k:v for k, v in save_model.items() if k in model_dict}    # 找出在当前网络中的参数
            model_dict.update(state_dict)
            cap.load_state_dict(model_dict) 
            print("Model loaded from '%s'"%(path))
    
    logdir = '../logs/'
    if os.path.exists(logdir) and del_dir:
        shutil.rmtree(logdir)
    time_stamp = "{0:%Y-%m-%d/%H-%M-%S}-epoch{1}/".format(datetime.now(), epochs)
    writer = SummaryWriter(log_dir = logdir+time_stamp)
    batch_number = data_set.__len__()
    confmat = ConfusionMatrix(10)
    cap_opt = optim.Adam(cap.parameters(), lr = 1e-3)
    cap_sch = optim.lr_scheduler.ExponentialLR(cap_opt, gamma = gamma)
    torch.autograd.set_detect_anomaly(True)
    old_acc = 0.0
    acc_cnt = 0
    test_cnt = 0
    cap.eval()
    with torch.no_grad():
        for j, (bx, by) in enumerate(test_set):
            bx = bx.cuda()
            by = by.cuda()
            out, _ = cap(bx, by)
            confmat.addRawElement(by, out)
    # for i in range(epochs):
    #     for k, (bx, by) in enumerate(data_set):
    #         cap_opt.zero_grad()

    #         bx = Var(bx).cuda()
    #         by = Var(by).cuda()
    #         y_caps, reconstructs = cap(bx, by)
    #         margin_loss = margin_loss_func(y_caps, by) 
    #         recon_loss = recons_loss_func(reconstructs, bx.view(batch_size, -1))
    #         loss = margin_loss + ratio * recon_loss
    #         loss.backward()
            
    #         cap_opt.step()
    #         cap_sch.step()

    #         train_cnt = i * batch_number + k
    #         local_acc = MarginLoss.accCounter(y_caps, by) / batch_size
    #         acc = (local_acc + old_acc) / 2
    #         old_acc = local_acc
    #         writer.add_scalar('Loss/Total Loss', loss, train_cnt)
    #         writer.add_scalar('Loss/Reconstruction loss', recon_loss, train_cnt)
    #         writer.add_scalar('Loss/Capsule loss', margin_loss, train_cnt)
    #         writer.add_scalar('Acc/Train Set Accuracy', acc, train_cnt)

    #         if k % test_time == 0:
    #             cap.eval()
    #             eval_cnt = 0
    #             eval_total = len(test_set)
    #             with torch.no_grad():
    #                 for j, (bx, by) in enumerate(test_set):
    #                     bx = bx.cuda()
    #                     by = by.cuda()
    #                     out, _ = cap(bx, by)
    #                     eval_cnt += MarginLoss.accCounter(out, by)
    #                     if j >= 9: break
    #                     confmat.addRawElement(by, out)
    #                 test_acc = eval_cnt / (10.0 * batch_size)
    #             test_cnt += 1
    #             writer.add_scalar('Acc/Test Set Accuracy', test_acc, test_cnt)
    #             print("Epoch: %3d / %3d\t Batch %4d / %4d\t recons loss: %.4f\t total loss: %.4f\t acc: %.4f\t test acc: %.4f\t lr: %.4lf"%(
    #                 i, epochs, k, batch_number, recon_loss.item(), loss.item(), acc, test_acc, cap_sch.get_last_lr()[-1]
    #             ))
    #             cap.train()
    #         if k % save_time == 0:
    #             img_to_save = reconstructs.detach().view(batch_size, 1, 28, 28)
    #             save_image(img_to_save[:25], "..\\imgs\\G_%d.jpg"%(k + 1), nrow = 5, normalize = True)
    #         # break
    # writer.close()
    # torch.save({
    #     'model': cap.state_dict(),
    #     'optimizer': cap_opt.state_dict()},
    #     path
    # )
    confmat.saveConfusionMatrix("..\\confusion.png")
    print("Output completed.")
    