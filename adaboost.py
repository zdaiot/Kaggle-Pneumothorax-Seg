import numpy as np
import math
from solver import Train
from datasets.siim import get_loader
from tqdm import tqdm
import torch
import pickle


class AdaBoost(object):
    def __init__(self, boost_times, config, index, train_images, train_masks, val_images, val_masks, criterion):
        # 提升次数
        self.boost_times = boost_times
        self.samples_weight = None
        self.models_weight = torch.zeros(self.boost_times)
        self.current_error = 0

        # 数据集样本和掩膜的路径
        self.train_images = train_images
        self.train_masks = train_masks
        self.val_images = val_images
        self.val_masks = val_masks

        # 模型相关
        self.model = None
        # 样本的评价准则
        self.criterion = criterion

        self.config = config
        self.index = index
    
    def init_samples_weight(self):
        """得到初始化权重，平均权重
        """
        # 样本数目
        samples_num = len(self.train_images)
        average_weight = 1.0 / samples_num
        self.samples_weight = np.ones(samples_num) * average_weight
        self.samples_weight = torch.from_numpy(self.samples_weight).float()

    def train_model(self, boost_index):
        """完成一轮训练
        
        Args:
            config: 训练配置
            index: 训练的折数
        """
        train_loader, val_loader = get_loader(
            self.train_images, 
            self.train_masks, 
            self.samples_weight,
            self.val_images, 
            self.val_masks, 
            image_size=self.config.image_size_stage1,
            batch_size=self.config.batch_size_stage1,
            num_workers=self.config.num_workers,
            augmentation_flag=self.config.stage1_augmentation_flag
            )
        model_solver = Train(self.config, train_loader, val_loader)
        
        if not self.config.resume:
            # 唤醒状态下不训练第一阶段
            model_solver.train(self.index, boost_index)
        
        # 第二阶段训练
        train_loader, val_loader = get_loader(
            self.train_images, 
            self.train_masks, 
            self.samples_weight,
            self.val_images, 
            self.val_masks, 
            image_size=self.config.image_size_stage2,
            batch_size=self.config.batch_size_stage2,
            num_workers=self.config.num_workers,
            augmentation_flag=self.config.stage2_augmentation_flag
            )
        # 更新数据集
        model_solver.train_loader = train_loader
        model_solver.valid_loader = val_loader
        model_solver.train_stage2(self.index, boost_index)
        # 唤醒轮结束后，将resume置0
        self.config.resume = 0
        # 更新模型
        self.model = model_solver.unet
    
    def boost(self):
        """执行adaboost算法

        :return: 无
        """
        init_flag = True
        print('Start training from {} step.'.format(self.config.resume_boost_index))
        for boost_index in range(self.config.resume_boost_index, self.boost_times):
            print("Boosting: {} step...".format(boost_index))
            # 更新样本权重
            self.update_samples_weight(init_flag, boost_index-1)
            # 训练模型
            self.train_model(boost_index)
            # 更新分类误差率
            self.calculate_dataset_error()
            # 更新基学习器的权重
            self.update_model_weight(boost_index)
            init_flag = False
        # 保存最终的参数
        with open('adaboost.pkl', 'wb') as f:
            print('Saving models_weight and samples_weight...')
            pickle.dump([self.models_weight, self.samples_weight], f) 

    def update_samples_weight(self, init_flag=True, boost_index=0):
        """更新样本权重
        """
        print('Updating samples weight...')
        if init_flag:
            self.init_samples_weight()
        else:
            # 计算规范化因子，数据加载器的shuffle设置为false
            train_loader, _ = get_loader(
                self.train_images, 
                self.train_masks, 
                self.samples_weight,
                self.val_images, 
                self.val_masks, 
                image_size=self.config.image_size_stage2,
                batch_size=self.config.batch_size_stage2,
                num_workers=self.config.num_workers,
                augmentation_flag=False,
                shuffle=False
                )   
            self.model.eval()
            # 计算规范化因子
            z_m = 0
            with torch.no_grad():
                print('Calculating normalization factor...')
                right_num = 0
                tbar = tqdm(train_loader)
                for sample_index, (image, mask, sample_weight) in enumerate(tbar):
                    image = image.cuda()
                    mask = mask.cuda()
                    sample_weight = sample_weight.cuda()

                    pred = self.model(image)
                    pred_flat = pred.view(pred.size(0), -1)
                    pred_flat = (torch.sigmoid(pred_flat) > 0.5).float()
                    mask_flat = mask.view(mask.size(0), -1)
                    dice = self.criterion(pred_flat, mask_flat)
                    # 预测正确和预测错误的样本分开计算
                    right_id = dice > 0.75
                    z_m += (sample_weight[right_id] * torch.exp(-1 * self.models_weight[boost_index])).sum()
                    z_m += (sample_weight[1 - right_id] * torch.exp(-1 * self.models_weight[boost_index] * -1)).sum()
                    
                    right_num += right_id.sum().item()
                    descript = 'Right detection samples num:%d, z_m: %.5f' % (right_num, z_m)
                    tbar.set_description(desc=descript)
                print("Calculating samples weight...")                
                right_num = 0
                # 更新样本权重，一定要按照顺序更新，样本与其权重要相对应
                seen_samples_num = 0
                tbar = tqdm(train_loader)
                for sample_index, (image, mask, sample_weight) in enumerate(tbar):
                    image = image.cuda()
                    mask = mask.cuda()
                    sample_weight = sample_weight.cuda()
                    pred = self.model(image)
                    pred_flat = pred.view(pred.size(0), -1)
                    pred_flat = (torch.sigmoid(pred_flat) > 0.5).float()
                    mask_flat = mask.view(mask.size(0), -1)
                    dice = self.criterion(pred_flat, mask_flat)
                    right_id = dice > 0.75
                    sample_weight[right_id] = sample_weight[right_id] / (z_m * torch.exp(-1 * self.models_weight[boost_index]))
                    sample_weight[1 - right_id] = sample_weight[1 - right_id] / (z_m * torch.exp(-1 * self.models_weight[boost_index] * -1))

                    self.samples_weight[seen_samples_num : seen_samples_num + sample_weight.size(0)] = sample_weight
                    seen_samples_num += sample_weight.size(0)
                    
                    right_num += right_id.sum().item()
                    descript = 'right / seen_samples: %d / %d' % (right_num, seen_samples_num)
                    tbar.set_description(desc=descript)

    def update_model_weight(self, boost_index):
        """更新基学习器的权重
        """
        print('Updating model weight...')
        if self.current_error >= 1:
            self.current_error = 1 - 1e-3
        elif self.current_error <= 0:
            self.current_error = 1e-3
        alpha = 1/2 * math.log((1-self.current_error) / self.current_error)
        self.models_weight[boost_index] = alpha
    
    def calculate_dataset_error(self):
        """计算模型在加权训练数据集上的分类误差率
        """
        print('Calculate dataset error...')
        train_loader, _ = get_loader(
            self.train_images, 
            self.train_masks, 
            self.samples_weight,
            self.val_images, 
            self.val_masks, 
            image_size=self.config.image_size_stage2,
            batch_size=self.config.batch_size_stage2,
            num_workers=self.config.num_workers,
            augmentation_flag=False
            )

        tbar = tqdm(train_loader)
        loss_weight_sum = 0
        self.model.eval()
        with torch.no_grad():
            for sample_index, (image, mask, sample_weight) in enumerate(tbar):
                image = image.cuda()
                mask = mask.cuda()
                sample_weight = sample_weight.cuda()
                pred = self.model(image)
                pred_flat = pred.view(pred.size(0), -1)
                pred_flat = (torch.sigmoid(pred_flat) > 0.5).float()
                mask_flat = mask.view(mask.size(0), -1)
                dice = self.criterion(pred_flat, mask_flat)
                descript = ' Dice mean: %.5f' % dice.mean().item()
                tbar.set_description(desc=descript)

                sample_weight[dice > 0.75] = 0.0
                loss_weight_sum += sample_weight.sum().item()
                descript = 'Loss weight sum: %.5f' % loss_weight_sum + descript
                tbar.set_description(desc=descript)

        print('Current_error: %.4f'%loss_weight_sum)
        self.current_error = loss_weight_sum