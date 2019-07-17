import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from utils.mask_functions import write_txt
from models.network import U_Net, R2U_Net, AttU_Net, R2AttU_Net
import csv
import matplotlib.pyplot as plt
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
        self.model_type = config.model_type
        self.t = config.t

        self.mode = config.mode
        self.resume = config.resume
        self.num_epochs_decay = config.num_epochs_decay

        # Hyper-parameters
        self.augmentation_prob = config.augmentation_prob
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # save set
        self.save_step = config.save_step
        self.save_path = config.save_path

        # 配置参数
        self.two_stage = config.two_stage
        self.epoch_stage1 = config.epoch_stage1
        self.epoch_stage1_freeze = config.epoch_stage1_freeze
        self.epoch_stage2 = config.epoch_stage2
        self.epoch_stage2_accumulation = config.epoch_stage2_accumulation
        self.accumulation_steps = config.accumulation_steps

        # 模型初始化
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.build_model()

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
            self.unet = Unet(backbone_name='resnet34', pretrained=True, classes=self.output_ch)

        if torch.cuda.is_available():
            self.unet = torch.nn.DataParallel(self.unet)
            self.criterion = self.criterion.cuda()
        self.unet.to(self.device)

        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                    self.lr, [self.beta1, self.beta2]) # TODO

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

    def freeze_encoder(self, epoch):
        for param in self.unet.module.backbone.parameters():
            param.requires_grad = False
        print('Stage1 epoch:{} freeze encoder'.format(epoch))
        write_txt(self.save_path, 'Stage1 epoch:{} freeze encoder'.format(epoch))

    def unfreeze_encoder(self, epoch):
        for param in self.unet.module.backbone.parameters():
            param.requires_grad = True
        print('Stage1 epoch:{} unfreeze encoder'.format(epoch))
        write_txt(self.save_path, 'Stage1 epoch:{} unfreeze encoder'.format(epoch))

    def train(self):
        if self.resume:
            weight_path = os.path.join(self.save_path, self.resume)
            if os.path.isfile(weight_path):
                # Load the pretrained Encoder
                if torch.cuda.is_available:
                    self.unet.module.load_state_dict(torch.load(weight_path))
                else:
                    self.unet.load_state_dict(torch.load(weight_path))
                print('Stage1 %s is Successfully Loaded from %s' % (self.model_type, weight_path))
                write_txt(self.save_path, 'Stage1 %s is Successfully Loaded from %s' % (self.model_type, weight_path))
            else:
                raise FileNotFoundError("Can not find weight file in {}".format(weight_path))

        # Train for Encoder
        lr = self.lr
        for epoch in range(self.epoch_stage1):
            epoch += 1
            self.unet.train(True)
            if epoch < self.epoch_stage1_freeze:
                self.freeze_encoder(epoch)
            else:
                self.unfreeze_encoder(epoch)
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
            print('Stage1 Epoch [%d/%d], Loss: %.5f' % (epoch, self.epoch_stage1, epoch_loss/len(tbar)))
            write_txt(self.save_path, 'Stage1 Epoch [%d/%d], Loss: %.5f' % (epoch, self.epoch_stage1, epoch_loss/len(tbar)))

            self.validation()

            if epoch % self.save_step == 0:
                pth_path = os.path.join(self.save_path, '%s_%d.pth' % (self.model_type, epoch))
                print('Saving Model.')
                write_txt(self.save_path, 'Saving Model.')
                if torch.cuda.is_available():
                    torch.save(self.unet.module.state_dict(), pth_path)
                else:
                    torch.save(self.unet.state_dict(), pth_path)

            # Decay learning rate
            if epoch > (self.epoch_stage1 + self.epoch_stage2 - self.num_epochs_decay):
                lr -= (self.lr / float(self.num_epochs_decay))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                print('Decay learning rate to lr: {}.'.format(lr))
                write_txt(self.save_path, 'Decay learning rate to lr: {}.'.format(lr))
        # 重新调整对象的lr，为train_stage2做准备
        self.lr = lr 

    def train_stage2(self):
        weight_path = os.path.join(self.save_path, '%s_%d.pth' % (self.model_type, 229)) # self.epoch_stage1
        if os.path.isfile(weight_path):
            # Load the pretrained Encoder
            if torch.cuda.is_available:
                self.unet.module.load_state_dict(torch.load(weight_path))
            else:
                self.unet.load_state_dict(torch.load(weight_path))
            print('Stage2: %s is Successfully Loaded from %s' % (self.model_type, weight_path))
            write_txt(self.save_path, 'Stage2: %s is Successfully Loaded from %s' % (self.model_type, weight_path))
        else:
            raise FileNotFoundError("Can not find weight file in {}".format(weight_path))

        # Train for Encoder
        lr = self.lr
        for epoch in range(self.epoch_stage2):
            epoch += 1
            self.unet.train(True)
            epoch_loss = 0

            self.reset_grad() # 梯度累加的时候需要使用
            
            tbar = tqdm.tqdm(self.train_loader)
            for i, (images, masks) in enumerate(tbar):
                # GT : Ground Truth
                images = images.to(self.device)
                masks = masks.to(self.device)
                assert images.size(2) == 1024

                # SR : Segmentation Result
                net_output = self.unet(images)
                net_output_flat = net_output.view(net_output.size(0), -1)
                masks_flat = masks.view(masks.size(0), -1)
                loss = self.criterion(net_output_flat, masks_flat)
                epoch_loss += loss.item()

                # Backprop + optimize, see https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/20 for Accumulating Gradients
                if epoch <= self.epoch_stage2 - self.epoch_stage2_accumulation:
                    self.reset_grad()
                    loss.backward()
                    self.optimizer.step()
                else:
                    # loss = loss / self.accumulation_steps                # Normalize our loss (if averaged)
                    loss.backward()                                      # Backward pass
                    if (i+1) % self.accumulation_steps == 0:             # Wait for several backward steps
                        self.optimizer.step()                            # Now we can do an optimizer step
                        self.reset_grad()

                descript = "Train Loss: %.5f" % (epoch_loss / (i + 1))
                tbar.set_description(desc=descript)

            # Print the log info
            print('Stage2 Epoch [%d/%d], Loss: %.5f' % (epoch, self.epoch_stage2, epoch_loss/len(tbar)))
            write_txt(self.save_path, 'Stage2 Epoch [%d/%d], Loss: %.5f' % (epoch, self.epoch_stage2, epoch_loss/len(tbar)))

            self.validation()

            if epoch % self.save_step == 0:
                pth_path = os.path.join(self.save_path, '%s_%d.pth' % (self.model_type, epoch+self.epoch_stage1))
                print('Saving Model.')
                write_txt(self.save_path, 'Saving Model.')
                if torch.cuda.is_available():
                    torch.save(self.unet.module.state_dict(), pth_path)
                else:
                    torch.save(self.unet.state_dict(), pth_path)

            # Decay learning rate
            if (self.epoch_stage1 + epoch) > (self.epoch_stage1 + self.epoch_stage2 - self.num_epochs_decay):
                lr -= (self.lr / float(self.num_epochs_decay))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                print('Decay learning rate to lr: {}.'.format(lr))
                write_txt(self.save_path, 'Decay learning rate to lr: {}.'.format(lr))

    def validation(self):
        self.unet.train(False)
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

        print('Val Loss: %.5f' % (loss_sum/len(tbar)))
        write_txt(self.save_path, 'Val Loss: %.5f' % (loss_sum/len(tbar)))

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
        print('Loaded from %s' % model_path)
        self.unet.train(False)
        self.unet.eval()
        # 先大概选取范围
        dices_ = []
        thrs_ = np.arange(0.1, 1, 0.1)  # 阈值列表
        for th in thrs_:
            tmp = []
            tbar = tqdm.tqdm(self.train_loader)
            for i, (images, masks) in enumerate(tbar):
                # GT : Ground Truth
                images = images.to(self.device)
                net_output = torch.nn.functional.sigmoid(self.unet(images))
                preds = (net_output > th).to(self.device).float()  # 大于阈值的归为1
                # preds[preds.view(preds.shape[0],-1).sum(-1) < noise_th,...] = 0.0 # 过滤噪声点
                tmp.append(self.dice_overall(preds, masks).mean())
            dices_.append(sum(tmp) / len(tmp))
        dices_ = np.array(dices_)
        best_thrs_ = thrs_[dices_.argmax()]
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

        plt.subplot(1, 2, 1)
        plt.title('Large-scale search')
        plt.plot(thrs_, dices_)
        plt.subplot(1, 2, 2)
        plt.title('Little-scale search')
        plt.plot(thrs, dices)
        plt.show()
        return best_thrs