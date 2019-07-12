import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from utils.evaluation import *
from models.network import U_Net, R2U_Net, AttU_Net, R2AttU_Net
import csv
import tqdm
from backboned_unet import Unet


class Train(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):

        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.augmentation_prob = config.augmentation_prob

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        # Step size
        self.save_step = config.save_step

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode
        self.resume = config.resume

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.t = config.t
        self.build_model()

        if not os.path.exists(os.path.join(self.model_path, self.model_type)):
            print('Making pth folder...')
            os.mkdir(os.path.join(self.model_path, self.model_type))

    def build_model(self):
        """Build generator and discriminator."""
        if self.model_type == 'U_Net':
            self.unet = U_Net(img_ch=3, output_ch=self.output_ch)
        elif self.model_type == 'R2U_Net':
            self.unet = R2U_Net(img_ch=3, output_ch=self.output_ch, t=self.t)
        elif self.model_type == 'AttU_Net':
            self.unet = AttU_Net(img_ch=3, output_ch=self.output_ch)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=3, output_ch=self.output_ch, t=self.t)
        elif self.model_type == 'unet_resnet34':
            self.unet = Unet(backbone_name='resnet34', pretrained=False, classes=1)

        if torch.cuda.is_available():
            self.unet = torch.nn.DataParallel(self.unet)
            self.criterion = self.criterion.cuda()
        self.unet.to(self.device)

        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                    self.lr, [self.beta1, self.beta2])

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def train(self):
        if self.resume:
            weight_path = os.path.join(self.model_path, self.model_type, self.resume)
            if os.path.isfile(weight_path):
                """Train encoder, generator and discriminator."""
                # Load the pretrained Encoder
                if torch.cuda.is_available:
                    self.unet.module.load_state_dict(torch.load(weight_path))
                else:
                    self.unet.load_state_dict(torch.load(weight_path))
                print('%s is Successfully Loaded from %s' % (self.model_type, weight_path))
            else:
                raise FileNotFoundError("Can not find weight file in {}".format(weight_path))

        # Train for Encoder
        lr = self.lr
        for epoch in range(self.num_epochs):
            self.unet.train(True)
            epoch_loss = 0
            tbar = tqdm.tqdm(self.train_loader)
            for i, (images, masks) in enumerate(tbar):
                # GT : Ground Truth
                images = images.to(self.device)
                masks = masks.to(self.device)

                # SR : Segmentation Result
                net_output = self.unet(images)
                net_output_flat = net_output.view(net_output.size(0), -1)
                masks_flat = masks.view(masks.size(0), -1)
                loss = self.criterion(net_output_flat, masks_flat)
                epoch_loss += loss.item()

                # Backprop + optimize
                self.reset_grad()
                loss.backward()
                self.optimizer.step()

                descript = "Train Loss: %.5f" % (epoch_loss / (i + 1))
                tbar.set_description(desc=descript)

            # Print the log info
            print('Epoch [%d/%d], Loss: %.5f' % (epoch + 1, self.num_epochs, epoch_loss))

            self.validation()

            if (epoch+1)%self.save_step == 0:
                pth_path = os.path.join(self.model_path, self.model_type, '%s_%d.pth' % (self.model_type, epoch))
                print('Saving Model.')
                if torch.cuda.is_available():
                    torch.save(self.unet.module.state_dict(), pth_path)
                else:
                    torch.save(self.unet.state_dict(), pth_path)

            # Decay learning rate
            if (epoch + 1) > (self.num_epochs - self.num_epochs_decay):
                lr -= (self.lr / float(self.num_epochs_decay))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                print('Decay learning rate to lr: {}.'.format(lr))

    def validation(self):
        print('Start Validatate:')

        tbar = tqdm.tqdm(self.valid_loader)
        loss_sum = 0
        for i, (images, masks) in enumerate(tbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            net_output = self.unet(images)
            net_output_flat = net_output.view(net_output.size(0), -1)
            masks_flat = masks.view(masks.size(0), -1)
            loss = self.criterion(net_output_flat, masks_flat)
            loss_sum += loss.item()

            descript = "Val Loss: %.5f" % (loss_sum / (i + 1))
            tbar.set_description(desc=descript)

        print('Val Loss: %.5f' % (loss_sum))

    # dice for threshold selection
    def dice_overall(self, preds, targs):
        n = preds.shape[0]  # batch size为多少
        preds = preds.view(n, -1)
        targs = targs.view(n, -1)
        preds, targs = preds.to(self.device), targs.to(self.device)
        # tensor之间按位相成，求两个集合的交(只有1×1等于1)后。按照第二个维度求和，得到[batch size]大小的tensor，每一个值代表该输入图片真实类标与预测类标的交集大小
        intersect = (preds * targs).sum(-1).float()
        # tensor之间按位相加，求两个集合的并。然后按照第二个维度求和，得到[batch size]大小的tensor，每一个值代表该输入图片真实类标与预测类标的并集大小
        union = (preds + targs).sum(-1).float()
        u0 = union == 0  # 寻找输入图片真实类标与预测类标无并集的情况
        intersect[u0] = 1
        union[u0] = 2 # 让无并集的位置置为2，防止分母为0
        return (2. * intersect / union)

    def choose_threshold(self, model_path, noise_th=75.0 * (256 / 128.0) ** 2):
        self.unet.module.load_state_dict(torch.load(model_path))
        self.unet.train(False)
        self.unet.eval()
        # 先大概选取范围
        dices = []
        thrs = np.arange(0.1, 1, 0.1)  # 阈值列表
        for th in thrs:
            tmp = []
            tbar = tqdm.tqdm(self.train_loader)
            for i, (images, masks) in enumerate(tbar):
                # GT : Ground Truth
                images = images.to(self.device)
                net_output = torch.nn.functional.sigmoid(self.unet(images))
                preds = (net_output > th).to(self.device).float()  # 大于阈值的归为1
                # preds[preds.view(preds.shape[0],-1).sum(-1) < noise_th,...] = 0.0 # 过滤噪声点
                tmp.append(self.dice_overall(preds, masks).mean())
            dices.append(sum(tmp) / len(tmp))
        dices = np.array(dices)
        best_thrs_ = thrs[dices.argmax()]
        # 精细选取范围
        dices = []
        thrs = np.arange(best_thrs_-0.05, best_thrs_+0.05, 0.01)  # 阈值列表
        for th in thrs:
            tmp = []
            tbar = tqdm.tqdm(self.train_loader)
            for i, (images, masks) in enumerate(tbar):
                # GT : Ground Truth
                images = images.to(self.device)
                net_output = torch.nn.functional.sigmoid(self.unet(images))
                preds = (net_output > th).to(self.device).float()  # 大于阈值的归为1
                # preds[preds.view(preds.shape[0],-1).sum(-1) < noise_th,...] = 0.0 # 过滤噪声点
                tmp.append(self.dice_overall(preds, masks).mean())
            dices.append(sum(tmp) / len(tmp))
        dices = np.array(dices)
        best_thrs = thrs[dices.argmax()]
        print('best_thrs:', best_thrs)
        return best_thrs