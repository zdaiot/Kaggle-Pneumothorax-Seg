import os, shutil
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
from models.linknet import LinkNet34
from models.deeplabv3.deeplabv3plus import DeepLabV3Plus
import csv
import matplotlib.pyplot as plt
import tqdm
from backboned_unet import Unet
from utils.loss import GetLoss, RobustFocalLoss2d, BCEDiceLoss, SoftBCEDiceLoss, SoftBceLoss, LovaszLoss
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp
from models.Transpose_unet.unet.model import Unet as Unet_t
from models.octave_unet.unet.model import OctaveUnet


class Train(object):
    def __init__(self, config, train_loader, valid_loader):
        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.criterion = SoftBCEDiceLoss(weight=[0.25, 0.75])
        # self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(50))
        self.criterion_stage2 = SoftBCEDiceLoss(weight=[0.25, 0.75])
        self.criterion_stage3 = SoftBCEDiceLoss(weight=[0.25, 0.75])
        self.model_type = config.model_type
        self.t = config.t

        self.mode = config.mode
        self.resume = config.resume

        # Hyper-parameters
        self.lr = config.lr
        self.lr_stage2 = config.lr_stage2
        self.lr_stage3 = config.lr_stage3
        self.start_epoch, self.max_dice = 0, 0
        self.weight_decay = config.weight_decay
        self.weight_decay_stage2 = config.weight_decay
        self.weight_decay_stage3 = config.weight_decay

        # save set
        self.save_path = config.save_path
        if 'choose_threshold' not in self.mode:
            TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
            self.writer = SummaryWriter(log_dir=self.save_path+'/'+TIMESTAMP)

        # 配置参数
        self.epoch_stage1 = config.epoch_stage1
        self.epoch_stage1_freeze = config.epoch_stage1_freeze
        self.epoch_stage2 = config.epoch_stage2
        self.epoch_stage2_accumulation = config.epoch_stage2_accumulation
        self.accumulation_steps = config.accumulation_steps
        self.epoch_stage3 = config.epoch_stage3
        self.epoch_stage3_accumulation = config.epoch_stage3_accumulation

        # 模型初始化
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.build_model()

    def build_model(self):
        print("Using model: {}".format(self.model_type))
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
            # self.unet = Unet(backbone_name='resnet34', pretrained=True, classes=self.output_ch)
            self.unet = smp.Unet('resnet34', encoder_weights='imagenet', activation=None)
        elif self.model_type == 'unet_resnet50':
            self.unet = smp.Unet('resnet50', encoder_weights='imagenet', activation=None)
        elif self.model_type == 'unet_se_resnext50_32x4d':
            self.unet = smp.Unet('se_resnext50_32x4d', encoder_weights='imagenet', activation=None)
        elif self.model_type == 'unet_densenet121':
            self.unet = smp.Unet('densenet121', encoder_weights='imagenet', activation=None)
        elif self.model_type == 'unet_resnet34_t':
            self.unet = Unet_t('resnet34', encoder_weights='imagenet', activation=None, use_ConvTranspose2d=True)
        elif self.model_type == 'unet_resnet34_oct':
            self.unet = OctaveUnet('resnet34', encoder_weights='imagenet', activation=None)
        
        elif self.model_type == 'linknet':
            self.unet = LinkNet34(num_classes=self.output_ch)
        elif self.model_type == 'deeplabv3plus':
            self.unet = DeepLabV3Plus(model_backbone='res50_atrous', num_classes=self.output_ch)
        elif self.model_type == 'pspnet_resnet34':
            self.unet = smp.PSPNet('resnet34', encoder_weights='imagenet', classes=1, activation=None)

        if torch.cuda.is_available():
            self.unet = torch.nn.DataParallel(self.unet)
            self.criterion = self.criterion.cuda()
            self.criterion_stage2 = self.criterion_stage2.cuda()
            self.criterion_stage3 = self.criterion_stage3.cuda()
        self.unet.to(self.device)

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

    def save_checkpoint(self, state, stage, index, is_best): 
        # 保存权重，每一epoch均保存一次，若为最优，则复制到最优权重；index可以区分不同的交叉验证 
        pth_path = os.path.join(self.save_path, '%s_%d_%d.pth' % (self.model_type, stage, index))
        torch.save(state, pth_path)
        if is_best:
            print('Saving Best Model.')
            write_txt(self.save_path, 'Saving Best Model.')
            shutil.copyfile(os.path.join(self.save_path, '%s_%d_%d.pth' % (self.model_type, stage, index)), os.path.join(self.save_path, '%s_%d_%d_best.pth' % (self.model_type, stage, index)))

    def load_checkpoint(self, load_optimizer=True):
        # Load the pretrained Encoder
        weight_path = os.path.join(self.save_path, self.resume)
        if os.path.isfile(weight_path):
            checkpoint = torch.load(weight_path)
            # 加载模型的参数，学习率，优化器，开始的epoch，最小误差等
            if torch.cuda.is_available:
                self.unet.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.unet.load_state_dict(checkpoint['state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.max_dice = checkpoint['max_dice']
            if load_optimizer:
                self.lr = checkpoint['lr']
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            print('%s is Successfully Loaded from %s' % (self.model_type, weight_path))
            write_txt(self.save_path, '%s is Successfully Loaded from %s' % (self.model_type, weight_path))
        else:
            raise FileNotFoundError("Can not find weight file in {}".format(weight_path))

    def train(self, index):
        # self.optimizer = optim.Adam([{'params': self.unet.decoder.parameters(), 'lr': 1e-4}, {'params': self.unet.encoder.parameters(), 'lr': 1e-6},])
        self.optimizer = optim.Adam(self.unet.module.parameters(), self.lr, weight_decay=self.weight_decay)

        # 若训练到一半暂停了，则需要加载之前训练的参数，并加载之前学习率 TODO:resume学习率没有接上，所以resume暂时无法使用
        if self.resume:
            self.load_checkpoint(load_optimizer=True)
            '''
            CosineAnnealingLR：若存在['initial_lr']，则从initial_lr开始衰减；
            若不存在，则执行CosineAnnealingLR会在optimizer.param_groups中添加initial_lr键值，其值等于lr

            重置初始学习率，在load_checkpoint中会加载优化器，但其中的initial_lr还是之前的，所以需要覆盖为self.lr，让其从self.lr衰减
            '''
            self.optimizer.param_groups[0]['initial_lr'] = self.lr
            
        stage1_epoches = self.epoch_stage1 - self.start_epoch
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, stage1_epoches+5)
        # 防止训练到一半暂停重新训练，日志被覆盖
        global_step_before = self.start_epoch*len(self.train_loader)

        for epoch in range(self.start_epoch, self.epoch_stage1):
            epoch += 1
            self.unet.train(True)
            
            # 学习率重启
            # if epoch == 30:
            #     self.optimizer.param_groups[0]['initial_lr'] = 0.0001
            #     lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 25)

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
                loss_set = self.criterion(net_output_flat, masks_flat)
                
                try:
                    loss_num = len(loss_set)
                except:
                    loss_num = 1
                # 依据返回的损失个数分情况处理
                if loss_num > 1:
                    for loss_index, loss_item in enumerate(loss_set):
                        if loss_index > 0:
                            loss_name = 'stage1_loss_%d' % loss_index
                            self.writer.add_scalar(loss_name, loss_item.item(), global_step_before + i)
                    loss = loss_set[0]
                else:
                    loss = loss_set
                epoch_loss += loss.item()

                # Backprop + optimize
                self.reset_grad()
                loss.backward()
                self.optimizer.step()
                
                params_groups_lr = str()
                for group_ind, param_group in enumerate(self.optimizer.param_groups):
                    params_groups_lr = params_groups_lr + 'params_group_%d' % (group_ind) + ': %.12f, ' % (param_group['lr'])

                # 保存到tensorboard，每一步存储一个
                self.writer.add_scalar('Stage1_train_loss', loss.item(), global_step_before+i)

                descript = "Train Loss: %.7f, lr: %s" % (loss.item(), params_groups_lr)
                tbar.set_description(desc=descript)
            # 更新global_step_before为下次迭代做准备
            global_step_before += len(tbar)

            # Print the log info
            print('Finish Stage1 Epoch [%d/%d], Average Loss: %.7f' % (epoch, self.epoch_stage1, epoch_loss/len(tbar)))
            write_txt(self.save_path, 'Finish Stage1 Epoch [%d/%d], Average Loss: %.7f' % (epoch, self.epoch_stage1, epoch_loss/len(tbar)))

            # 验证模型，保存权重，并保存日志
            loss_mean, dice_mean = self.validation(stage=1)
            if dice_mean > self.max_dice: 
                is_best = True
                self.max_dice = dice_mean
            else: is_best = False
            
            self.lr = lr_scheduler.get_lr()
            state = {'epoch': epoch,
                'state_dict': self.unet.module.state_dict(),
                'max_dice': self.max_dice,
                'optimizer' : self.optimizer.state_dict(),
                'lr' : self.lr}
            
            self.save_checkpoint(state, 1, index, is_best)

            self.writer.add_scalar('Stage1_val_loss', loss_mean, epoch)
            self.writer.add_scalar('Stage1_val_dice', dice_mean, epoch)
            self.writer.add_scalar('Stage1_lr', self.lr[0], epoch)

            # 学习率衰减
            lr_scheduler.step()

    def train_stage2(self, index):
        # # 冻结BN层， see https://zhuanlan.zhihu.com/p/65439075 and https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/100736591271 for more information
        # def set_bn_eval(m):
        #     classname = m.__class__.__name__
        #     if classname.find('BatchNorm') != -1:
        #         m.eval()
        # self.unet.apply(set_bn_eval)

        # self.optimizer = optim.Adam([{'params': self.unet.decoder.parameters(), 'lr': 1e-5}, {'params': self.unet.encoder.parameters(), 'lr': 1e-7},])
        self.optimizer = optim.Adam(self.unet.module.parameters(), self.lr_stage2, weight_decay=self.weight_decay_stage2)

        # 加载的resume分为两种情况：之前没有训练第二个阶段，现在要加载第一个阶段的参数；第二个阶段训练了一半要继续训练
        if self.resume:
            # 若第二个阶段训练一半，要重新加载 TODO
            if self.resume.split('_')[2] == '2':
                self.load_checkpoint(load_optimizer=True) # 当load_optimizer为True会重新加载学习率和优化器
                '''
                CosineAnnealingLR：若存在['initial_lr']，则从initial_lr开始衰减；
                若不存在，则执行CosineAnnealingLR会在optimizer.param_groups中添加initial_lr键值，其值等于lr

                重置初始学习率，在load_checkpoint中会加载优化器，但其中的initial_lr还是之前的，所以需要覆盖为self.lr，让其从self.lr衰减
                '''
                self.optimizer.param_groups[0]['initial_lr'] = self.lr

            # 若第一阶段结束后没有直接进行第二个阶段，中间暂停了
            elif self.resume.split('_')[2] == '1':
                self.load_checkpoint(load_optimizer=False)
                self.start_epoch = 0
                self.max_dice = 0

        # 第一阶段结束后直接进行第二个阶段，中间并没有暂停
        else:
            self.start_epoch = 0
            self.max_dice = 0

        # 防止训练到一半暂停重新训练，日志被覆盖
        global_step_before = self.start_epoch*len(self.train_loader)

        stage2_epoches = self.epoch_stage2 - self.start_epoch
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, stage2_epoches+5)

        for epoch in range(self.start_epoch, self.epoch_stage2):
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
                loss_set = self.criterion_stage2(net_output_flat, masks_flat)

                try:
                    loss_num = len(loss_set)
                except:
                    loss_num = 1
                # 依据返回的损失个数分情况处理
                if loss_num > 1:
                    for loss_index, loss_item in enumerate(loss_set):
                        if loss_index > 0:
                            loss_name = 'stage2_loss_%d' % loss_index
                            self.writer.add_scalar(loss_name, loss_item.item(), global_step_before + i)
                    loss = loss_set[0]
                else:
                    loss = loss_set
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

                params_groups_lr = str()
                for group_ind, param_group in enumerate(self.optimizer.param_groups):
                    params_groups_lr = params_groups_lr + 'params_group_%d' % (group_ind) + ': %.12f, ' % (param_group['lr'])

                # 保存到tensorboard，每一步存储一个
                self.writer.add_scalar('Stage2_train_loss', loss.item(), global_step_before+i)

                descript = "Train Loss: %.7f, lr: %s" % (loss.item(), params_groups_lr)
                tbar.set_description(desc=descript)
            # 更新global_step_before为下次迭代做准备
            global_step_before += len(tbar)

            # Print the log info
            print('Finish Stage2 Epoch [%d/%d], Average Loss: %.7f' % (epoch, self.epoch_stage2, epoch_loss/len(tbar)))
            write_txt(self.save_path, 'Finish Stage2 Epoch [%d/%d], Average Loss: %.7f' % (epoch, self.epoch_stage2, epoch_loss/len(tbar)))

            # 验证模型，保存权重，并保存日志
            loss_mean, dice_mean = self.validation(stage=2)
            if dice_mean > self.max_dice: 
                is_best = True
                self.max_dice = dice_mean
            else: is_best = False
            
            self.lr = lr_scheduler.get_lr()            
            state = {'epoch': epoch,
                'state_dict': self.unet.module.state_dict(),
                'max_dice': self.max_dice,
                'optimizer' : self.optimizer.state_dict(),
                'lr' : self.lr}
            
            self.save_checkpoint(state, 2, index, is_best)

            self.writer.add_scalar('Stage2_val_loss', loss_mean, epoch)
            self.writer.add_scalar('Stage2_val_dice', dice_mean, epoch)
            self.writer.add_scalar('Stage2_lr', self.lr[0], epoch)

            # 学习率衰减
            lr_scheduler.step()

    # stage3, 接着stage2的训练，只训练有mask的样本
    def train_stage3(self, index):
        # # 冻结BN层， see https://zhuanlan.zhihu.com/p/65439075 and https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/100736591271 for more information
        # def set_bn_eval(m):
        #     classname = m.__class__.__name__
        #     if classname.find('BatchNorm') != -1:
        #         m.eval()
        # self.unet.apply(set_bn_eval)

        # self.optimizer = optim.Adam([{'params': self.unet.decoder.parameters(), 'lr': 1e-5}, {'params': self.unet.encoder.parameters(), 'lr': 1e-7},])
        self.optimizer = optim.Adam(self.unet.module.parameters(), self.lr_stage3, weight_decay=self.weight_decay_stage3)

        # 如果是 train_stage23，则resume只在第二阶段起作用
        if self.mode == 'train_stage23':
            self.resume = None
        # 加载的resume分为两种情况：之前没有训练第三个阶段，现在要加载第二个阶段的参数；第三个阶段训练了一半要继续训练
        if self.resume:
            # 若第三个阶段训练一半，要重新加载 TODO
            if self.resume.split('_')[2] == '3':
                self.load_checkpoint(load_optimizer=True) # 当load_optimizer为True会重新加载学习率和优化器
                '''
                CosineAnnealingLR：若存在['initial_lr']，则从initial_lr开始衰减；
                若不存在，则执行CosineAnnealingLR会在optimizer.param_groups中添加initial_lr键值，其值等于lr

                重置初始学习率，在load_checkpoint中会加载优化器，但其中的initial_lr还是之前的，所以需要覆盖为self.lr，让其从self.lr衰减
                '''
                self.optimizer.param_groups[0]['initial_lr'] = self.lr

            # 若第二阶段结束后没有直接进行第三个阶段，中间暂停了
            elif self.resume.split('_')[2] == '2':
                self.load_checkpoint(load_optimizer=False)
                self.start_epoch = 0
                self.max_dice = 0

        # 第二阶段结束后直接进行第三个阶段，中间并没有暂停
        else:
            self.start_epoch = 0
            self.max_dice = 0

        # 防止训练到一半暂停重新训练，日志被覆盖
        global_step_before = self.start_epoch*len(self.train_loader)

        stage3_epoches = self.epoch_stage3 - self.start_epoch
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, stage3_epoches+5)

        for epoch in range(self.start_epoch, self.epoch_stage3):
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
                loss_set = self.criterion_stage3(net_output_flat, masks_flat)

                try:
                    loss_num = len(loss_set)
                except:
                    loss_num = 1
                # 依据返回的损失个数分情况处理
                if loss_num > 1:
                    for loss_index, loss_item in enumerate(loss_set):
                        if loss_index > 0:
                            loss_name = 'stage3_loss_%d' % loss_index
                            self.writer.add_scalar(loss_name, loss_item.item(), global_step_before + i)
                    loss = loss_set[0]
                else:
                    loss = loss_set
                epoch_loss += loss.item()

                # Backprop + optimize, see https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/20 for Accumulating Gradients
                if epoch <= self.epoch_stage3 - self.epoch_stage3_accumulation:
                    self.reset_grad()
                    loss.backward()
                    self.optimizer.step()
                else:
                    # loss = loss / self.accumulation_steps                # Normalize our loss (if averaged)
                    loss.backward()                                      # Backward pass
                    if (i+1) % self.accumulation_steps == 0:             # Wait for several backward steps
                        self.optimizer.step()                            # Now we can do an optimizer step
                        self.reset_grad()

                params_groups_lr = str()
                for group_ind, param_group in enumerate(self.optimizer.param_groups):
                    params_groups_lr = params_groups_lr + 'params_group_%d' % (group_ind) + ': %.12f, ' % (param_group['lr'])

                # 保存到tensorboard，每一步存储一个
                self.writer.add_scalar('Stage3_train_loss', loss.item(), global_step_before+i)

                descript = "Train Loss: %.7f, lr: %s" % (loss.item(), params_groups_lr)
                tbar.set_description(desc=descript)
            # 更新global_step_before为下次迭代做准备
            global_step_before += len(tbar)

            # Print the log info
            print('Finish Stage3 Epoch [%d/%d], Average Loss: %.7f' % (epoch, self.epoch_stage3, epoch_loss/len(tbar)))
            write_txt(self.save_path, 'Finish Stage3 Epoch [%d/%d], Average Loss: %.7f' % (epoch, self.epoch_stage3, epoch_loss/len(tbar)))

            # 验证模型，保存权重，并保存日志
            loss_mean, dice_mean = self.validation(stage=3)
            if dice_mean > self.max_dice: 
                is_best = True
                self.max_dice = dice_mean
            else: is_best = False
            
            self.lr = lr_scheduler.get_lr()            
            state = {'epoch': epoch,
                'state_dict': self.unet.module.state_dict(),
                'max_dice': self.max_dice,
                'optimizer' : self.optimizer.state_dict(),
                'lr' : self.lr}
            
            self.save_checkpoint(state, 3, index, is_best)

            self.writer.add_scalar('Stage3_val_loss', loss_mean, epoch)
            self.writer.add_scalar('Stage3_val_dice', dice_mean, epoch)
            self.writer.add_scalar('Stage3_lr', self.lr[0], epoch)

            # 学习率衰减
            lr_scheduler.step()
            

    def validation(self, stage=1):
        # 验证的时候，train(False)是必须的0，设置其中的BN层、dropout等为eval模式
        # with torch.no_grad(): 可以有，在这个上下文管理器中，不反向传播，会加快速度，可以使用较大batch size
        self.unet.eval()
        tbar = tqdm.tqdm(self.valid_loader)
        loss_sum, dice_sum = 0, 0
        if stage == 1:
            criterion = self.criterion
        elif stage == 2:
            criterion = self.criterion_stage2
        elif stage == 3:
            criterion = self.criterion_stage3
        with torch.no_grad(): 
            for i, (images, masks) in enumerate(tbar):
                images = images.to(self.device)
                masks = masks.to(self.device)

                net_output = self.unet(images)
                net_output_flat = net_output.view(net_output.size(0), -1)
                masks_flat = masks.view(masks.size(0), -1)
                
                loss_set = criterion(net_output_flat, masks_flat)
                try:
                    loss_num = len(loss_set)
                except:
                    loss_num = 1
                
                # 依据返回的损失个数分情况处理
                if loss_num > 1:
                    loss = loss_set[0]
                else:
                    loss = loss_set
                loss_sum += loss.item()

                # 计算dice系数，预测出的矩阵要经过sigmoid含义以及阈值，阈值默认为0.5
                net_output_flat_sign = (torch.sigmoid(net_output_flat)>0.5).float()
                dice = self.dice_overall(net_output_flat_sign, masks_flat).mean()
                dice_sum += dice.item()

                descript = "Val Loss: {:.7f}, dice: {:.7f}".format(loss.item(), dice.item())
                tbar.set_description(desc=descript)
        
        loss_mean, dice_mean = loss_sum/len(tbar), dice_sum/len(tbar)
        print("Val Loss: {:.7f}, dice: {:.7f}".format(loss_mean, dice_mean))
        write_txt(self.save_path, "Val Loss: {:.7f}, dice: {:.7f}".format(loss_mean, dice_mean))
        return loss_mean, dice_mean

    # dice for threshold selection
    def dice_overall(self, preds, targs):
        n = preds.shape[0]  # batch size为多少
        preds = preds.view(n, -1)
        targs = targs.view(n, -1)
        # preds, targs = preds.to(self.device), targs.to(self.device)
        preds, targs = preds.cpu(), targs.cpu()

        # tensor之间按位相成，求两个集合的交(只有1×1等于1)后。按照第二个维度求和，得到[batch size]大小的tensor，每一个值代表该输入图片真实类标与预测类标的交集大小
        intersect = (preds * targs).sum(-1).float()
        # tensor之间按位相加，求两个集合的并。然后按照第二个维度求和，得到[batch size]大小的tensor，每一个值代表该输入图片真实类标与预测类标的并集大小
        union = (preds + targs).sum(-1).float()
        '''
        输入图片真实类标与预测类标无并集有两种情况：第一种为预测与真实均没有类标，此时并集之和为0；第二种为真实有类标，但是预测完全错误，此时并集之和不为0;

        寻找输入图片真实类标与预测类标并集之和为0的情况，将其交集置为1，并集置为2，最后还有一个2*交集/并集，值为1；
        其余情况，直接按照2*交集/并集计算，因为上面的并集并没有减去交集，所以需要拿2*交集，其最大值为1
        '''
        u0 = union == 0
        intersect[u0] = 1
        union[u0] = 2
        
        return (2. * intersect / union)

    def choose_threshold(self, model_path, index):
        self.unet.module.load_state_dict(torch.load(model_path)['state_dict'])
        print('Loaded from %s' % model_path)
        self.unet.eval()
        
        with torch.no_grad():
            # 先大概选取阈值范围
            dices_big = []
            thrs_big = np.arange(0.1, 1, 0.1)  # 阈值列表
            for th in thrs_big:
                tmp = []
                tbar = tqdm.tqdm(self.valid_loader)
                for i, (images, masks) in enumerate(tbar):
                    # GT : Ground Truth
                    images = images.to(self.device)
                    net_output = torch.sigmoid(self.unet(images))
                    preds = (net_output > th).to(self.device).float()  # 大于阈值的归为1
                    # preds[preds.view(preds.shape[0],-1).sum(-1) < noise_th,...] = 0.0 # 过滤噪声点
                    tmp.append(self.dice_overall(preds, masks).mean())
                dices_big.append(sum(tmp) / len(tmp))
            dices_big = np.array(dices_big)
            best_thrs_big = thrs_big[dices_big.argmax()]

            # 精细选取范围
            dices_little = []
            thrs_little = np.arange(best_thrs_big-0.05, best_thrs_big+0.05, 0.01)  # 阈值列表
            for th in thrs_little:
                tmp = []
                tbar = tqdm.tqdm(self.valid_loader)
                for i, (images, masks) in enumerate(tbar):
                    # GT : Ground Truth
                    images = images.to(self.device)
                    net_output = torch.sigmoid(self.unet(images))
                    preds = (net_output > th).to(self.device).float()  # 大于阈值的归为1
                    # preds[preds.view(preds.shape[0],-1).sum(-1) < noise_th,...] = 0.0 # 过滤噪声点
                    tmp.append(self.dice_overall(preds, masks).mean())
                dices_little.append(sum(tmp) / len(tmp))
            dices_little = np.array(dices_little)
            # score = dices.max()
            best_thr = thrs_little[dices_little.argmax()]
            
            # 选最优像素阈值
            dices_pixel = []
            pixel_thrs = np.arange(0, 2304, 256)  # 阈值列表
            for pixel_thr in pixel_thrs:
                tmp = []
                tbar = tqdm.tqdm(self.valid_loader)
                for i, (images, masks) in enumerate(tbar):
                    # GT : Ground Truth
                    images = images.to(self.device)
                    net_output = torch.sigmoid(self.unet(images))
                    preds = (net_output > best_thr).to(self.device).float()  # 大于阈值的归为1
                    preds[preds.view(preds.shape[0],-1).sum(-1) < pixel_thr,...] = 0.0 # 过滤噪声点
                    tmp.append(self.dice_overall(preds, masks).mean())
                dices_pixel.append(sum(tmp) / len(tmp))
            dices_pixel = np.array(dices_pixel)
            score = dices_pixel.max()
            best_pixel_thr = pixel_thrs[dices_pixel.argmax()]
            print('best_thr:{}, best_pixel_thr:{}, score:{}'.format(best_thr, best_pixel_thr, score))

        plt.figure(figsize=(10.4, 4.8))
        plt.subplot(1, 3, 1)
        plt.title('Large-scale search')
        plt.plot(thrs_big, dices_big)
        plt.subplot(1, 3, 2)
        plt.title('Little-scale search')
        plt.plot(thrs_little, dices_little)
        plt.subplot(1, 3, 3)
        plt.title('pixel thrs search')
        plt.plot(pixel_thrs, dices_pixel)
        plt.savefig(os.path.join(self.save_path, str(index)+'png'))
        # plt.show()
        plt.close()
        return float(best_thr), float(best_pixel_thr), float(score)
