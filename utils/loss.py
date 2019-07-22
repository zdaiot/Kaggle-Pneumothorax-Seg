# 2019.7.16
import numpy as np
import time
import random

import torch
from torch import nn 
import torch.nn.functional as F
from torch.autograd import Variable


def one_hot(target, class_num):
    """进行one_hot编码
    Args:
        target: 原始类标
        class_num: 类别数目
    Return:
        target_oh: 经过编码的类标
    """
    assert target.dim() == 2 or target.dim() == 3

    origin_size = target.size()
    origin_numel = target.numel()
    target_flat = target.view(origin_size[0], -1)

    # target_oh的大小为[batch_size, class_num, 单个样本包含的像素数]
    target_oh = torch.zeros(origin_size[0], class_num, origin_numel/origin_size[0])

    for c in range(class_num):
        # 找出为第c类的像素
        target_index = target_flat == c
        # 将第c类中对应的这些像素置为1
        target_oh[:, c, :] = target_index
    
    if target.dim() > 2:
        target_oh =  target_oh.view(origin_size[0], class_num, origin_size[1], origin_size[2])
    else:
        target_oh = target_oh.view(origin_size[0], class_num)

    return target_oh


class DiceLoss(nn.Module):
    """二分类 dice loss
    """
    def __init__(self, sigmoid_flag=True):
        super(DiceLoss, self).__init__()
        self.sigmoid_flag = sigmoid_flag

    def forward(self, input, target):
        Num = input.size(0)
        smooth = 1 # smooth确保分母不为0，同时具有防止过拟合的作用

        if self.sigmoid_flag:
            input = torch.sigmoid(input)
        
        input_flat = input.contiguous().view(Num, -1)
        target_flat = target.contiguous().view(Num, -1)

        intersection = (input_flat * target_flat).sum(1)
        union = input_flat.sum(1) + target_flat.sum(1)
        loss = (2.0 * intersection + smooth) / (union + smooth)
        loss = 1 - loss.sum(0) / Num

        return loss


class MultiDiceLoss(nn.Module):
	"""多分类Dice损失	
	"""
	def __init__(self, class_num):
		super(MultiDiceLoss, self).__init__()
        self.class_num = class_num

	def forward(self, input, target, weights=None):
        
 		 
		dice = DiceLoss()
		totalLoss = 0
 
		for i in range(C):
			diceLoss = dice(input[:,i], target[:,i])
			if weights is not None:
				diceLoss *= weights[i]
			totalLoss += diceLoss
 
		return totalLoss


class FocalLoss(nn.Module):
    """二分类FocalLoss
    """
    def __init__(self, gamma=0, alpha=None, size_average=True):
        """
        Args:
            gamma: 聚焦因子
            alpha: 类别权重
            size_average: 是否在各样本之间取平均
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        """
        Args:
            input: 模型的预测，在二分类问题中，取sigmoid后，每一个元素表示对应样本属于正类的概率
            target: 对应样本的真实类标
        """
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        # 向input施加sigmod和log
        log_pt = F.logsigmoid(input)
        log_pt = log_pt * target # 筛选出正类对应的概率
        log_pt = log_pt.view(-1) # 变换为一维向量
        # 筛选出负类对应的概率
        log_pt_neg = (1-log_pt)  * (1-target)

        pt = Variable(log_pt.data.exp())
        pt_neg = Variable(log_pt_neg.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            log_pt = self.alpha * log_pt
            log_pt_neg = (1 - self.alpha) * log_pt_neg

        loss = -1 * (1-pt)**self.gamma*log_pt - 1 * (1-pt_neg)**self.gamma*log_pt_neg

        # 所有样本的损失取均值或求和
        if self.size_average: 
            return loss.mean()
        else:
            return loss.sum()


class MultiFocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(MultiFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        """
        Args:
            input: 模型的输入，取softmax后，表示对应样本属于各类的概率
            target: 真实类标
        """
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: 
            return loss.mean()
        else: 
            return loss.sum()


class GetLoss(nn.Module):
    """依据传入的损失函数列表，返回一个总的损失函数
    """
    def __init__(self, loss_funs, loss_weights, sigmoid_flag):
        """
        Args:
            loss_funs: list，需要使用的损失函数
            loss_weights: list，每一个损失对应的权重
        """
        super(GetLoss, self).__init__()
        self.loss_funs = torch.nn.ModuleList(loss_funs)
        self.loss_weights = loss_weights
        self.sigmoid_flag = sigmoid_flag
    
    def forward(self, input, target):
        if self.sigmoid_flag:
            input = torch.sigmoid(input)
        loss = 0
        for index, loss_fun in enumerate(self.loss_funs):
            loss += self.loss_weights[index] * loss_fun(input, target)

        return loss    
    

if __name__ == "__main__":
    start_time = time.time()
    maxe = 0
    for i in range(1000):
        x = torch.rand(12800,2)*random.randint(1,10)
        x = Variable(x.cuda())
        l = torch.rand(12800).ge(0.1).long()
        l = Variable(l.cuda())

        output0 = MultiFocalLoss(gamma=0)(x,l)
        output1 = nn.CrossEntropyLoss()(x,l)
        a = output0.item()
        b = output1.item()
        if abs(a-b)>maxe: maxe = abs(a-b)
    print('time:',time.time()-start_time,'max_error:',maxe)

    start_time = time.time()
    maxe = 0
    for i in range(100):
        x = torch.rand(128,1000,8,4)*random.randint(1,10)
        x = Variable(x.cuda())
        l = torch.rand(128,8,4)*1000    # 1000 is classes_num
        l = l.long()
        l = Variable(l.cuda())

        output0 = MultiFocalLoss(gamma=0)(x,l)
        output1 = nn.NLLLoss2d()(F.log_softmax(x),l)
        a = output0.item()
        b = output1.item()
        if abs(a-b) > maxe : maxe = abs(a-b)
    print('time:', time.time()-start_time,'max_error:',maxe)