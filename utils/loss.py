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
    target_oh = torch.zeros(origin_size[0], class_num, int(origin_numel/origin_size[0]))

    for c in range(class_num):
        # 找出真实标定中为第c类的样本
        target_index = target_flat == c
        # 将第c类中对应这些样本的位置 置为1
        target_oh[:, c, :] = target_index
    
    if target.dim() > 2:
        target_oh =  target_oh.view(origin_size[0], class_num, origin_size[1], origin_size[2])
    else:
        target_oh = target_oh.view(origin_size[0], class_num)

    return target_oh


def dice_overall(preds, targs):
    n = preds.shape[0]  # batch size为多少
    preds = preds.view(n, -1)
    targs = targs.view(n, -1)
    preds, targs = preds.cpu(), targs.cpu()

    # tensor之间按位相成，求两个集合的交(只有1×1等于1)后。按照第二个维度求和，得到[batch size]大小的tensor，每一个值代表该输入图片真实类标与预测类标的交集大小
    intersect = (preds * targs).sum(-1).float()
    # tensor之间按位相加，求两个集合的并。然后按照第二个维度求和，得到[batch size]大小的tensor，每一个值代表该输入图片真实类标与预测类标的并集大小
    union = (preds + targs).sum(-1).float()
    u0 = union == 0
    intersect[u0] = 1
    union[u0] = 2

    return 2. * intersect / union


# reference https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/101429#latest-588288
class SoftDiceLoss(nn.Module):
    """二分类加权dice损失
    """
    def __init__(self, size_average=True, weight=[0.2, 0.8]):
        """
        weight: 各类别权重
        """
        super(SoftDiceLoss, self).__init__()
        self.size_average = size_average
        self.weight = torch.FloatTensor(weight)
    
    def forward(self, logit_pixel, truth_pixel):
        batch_size = len(logit_pixel)
        logit = logit_pixel.view(batch_size, -1)
        truth = truth_pixel.view(batch_size, -1)
        assert(logit.shape == truth.shape)

        loss = self.soft_dice_criterion(logit, truth)

        if self.size_average:
            loss = loss.mean()
        return loss

    def soft_dice_criterion(self, logit, truth):
        batch_size = len(logit)
        probability = torch.sigmoid(logit)

        p = probability.view(batch_size, -1)
        t = truth.view(batch_size, -1)
        # 向各样本分配所属类别的权重
        w = truth.detach()
        self.weight = self.weight.type_as(logit)
        w = w * (self.weight[1] - self.weight[0]) + self.weight[0]

        p = w * (p*2 - 1)  #convert to [0,1] --> [-1, 1]
        t = w * (t*2 - 1)

        intersection = (p * t).sum(-1)
        union = (p * p).sum(-1) + (t * t).sum(-1)
        dice = 1 - 2 * intersection/union

        loss = dice
        return loss

# reference https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/101429#latest-588288
class SoftBceLoss(nn.Module):
    """二分类交叉熵加权损失
    """
    def __init__(self, weight=[0.25, 0.75]):
        super(SoftBceLoss, self).__init__()
        self.weight = weight

    def forward(self, logit_pixel, truth_pixel):
        logit = logit_pixel.view(-1)
        truth = truth_pixel.view(-1)
        assert(logit.shape==truth.shape)

        loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')
        if self.weight:
            pos = (truth>0.5).float()
            neg = (truth<0.5).float()
            # pos_weight = pos.sum().item() + 1e-12
            # neg_weight = neg.sum().item() + 1e-12
            # loss = (self.weight[0]*pos*loss/pos_weight + self.weight[1]*neg*loss/neg_weight).sum()
            loss = (self.weight[1]*pos*loss + self.weight[0]*neg*loss).mean()
        else:
            loss = loss.mean()
        return loss

class SoftBCEDiceLoss(nn.Module):
    """加权BCE+DiceLoss
    """
    def __init__(self, size_average=True, weight=[0.2, 0.8]):
        """
        weight: weight[0]为负类的权重，weight[1]为正类的权重
        """
        super(SoftBCEDiceLoss, self).__init__()
        self.size_average = size_average
        self.weight = weight
        if self.size_average:
            self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor(self.weight[1]))
        else:
            self.bce_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(self.weight[1]))
        # self.bce_loss = SoftBceLoss(weight=weight)
        self.softdiceloss = SoftDiceLoss(size_average=self.size_average, weight=weight)
    
    def forward(self, input, target, samples_weight):
        soft_bce_loss = self.bce_loss(input, target)
        soft_dice_loss = self.softdiceloss(input, target)
        soft_bce_loss = soft_bce_loss.mean(dim=1)
        loss = soft_bce_loss + soft_dice_loss
        loss = samples_weight * loss
        loss = loss.mean()

        return loss, soft_bce_loss.mean(), soft_dice_loss.mean()

class LovaszLoss(nn.Module):
    '''加权lovasz loss
    '''
    def __init__(self, margin=[1,5]):
        super(LovaszLoss, self).__init__()
        self.margin = margin

    def forward(self, logit_pixel, truth_pixel):
        batch_size = len(logit_pixel)
        logit = logit_pixel.view(batch_size,-1)
        truth = truth_pixel.view(batch_size,-1)
        assert(logit.shape==truth.shape)
    
        loss = self.lovasz_loss(logit, truth, self.margin)  
        loss = loss.mean()
        
        return loss

    def lovasz_loss(self, logit, truth, margin=[1,5]):

        def compute_lovasz_gradient(truth): #sorted
            truth_sum    = truth.sum()
            intersection = truth_sum - truth.cumsum(0)
            union        = truth_sum + (1 - truth).cumsum(0)
            jaccard      = 1. - intersection / union
            T = len(truth)
            jaccard[1:T] = jaccard[1:T] - jaccard[0:T-1]
    
            gradient = jaccard
            return gradient
    
        def lovasz_hinge_one(logit , truth):
    
            m = truth.detach()
            m = m*(margin[1]-margin[0])+margin[0]
    
            truth = truth.float()
            sign  = 2. * truth - 1.
            hinge = (m - logit * sign)
            hinge, permutation = torch.sort(hinge, dim=0, descending=True)
            hinge = F.relu(hinge)
    
            truth = truth[permutation.data]
            gradient = compute_lovasz_gradient(truth)
    
            loss = torch.dot(hinge, gradient)
            return loss
    
        #----
        lovasz_one = lovasz_hinge_one
    
        batch_size = len(truth)
        loss = torch.zeros(batch_size).cuda()
        for b in range(batch_size):
            l, t = logit[b].view(-1), truth[b].view(-1)
            loss[b] = lovasz_one(l, t)

        return loss


class MultiDiceLoss(nn.Module):
    """多分类Dice损失	
    """
    def __init__(self, class_num, weights):
        super(MultiDiceLoss, self).__init__()
        self.class_num = class_num
        self.weights = weights
        self.dice_loss = DiceLoss()

    def forward(self, input, target):
        # 对真实类标进行one-hot编码
        target_oh = one_hot(target, self.class_num)

        totalLoss = 0

        # 针对每一类分别计算Dice损失
        for i in range(self.class_num):
            diceLoss = self.dice_loss(input[:, i], target_oh[:, i])
            if self.weights is not None:
                diceLoss *= self.weights[i]
            totalLoss += diceLoss
 
        return totalLoss


class RobustFocalLoss2d(nn.Module):
    #assume top 10% is outliers
    def __init__(self, gamma=2, size_average=True):
        super(RobustFocalLoss2d, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit, target, class_weight=None, type='sigmoid'):
        target = target.view(-1, 1).long()

        if type=='sigmoid':
            if class_weight is None:
                class_weight = [1]*2 # [0.5, 0.5]

            prob = torch.sigmoid(logit)
            prob = prob.view(-1, 1)
            prob = torch.cat((1-prob, prob), 1)
            # select表示样本属于的类别(one-hot)形式
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)

        elif type == 'softmax':
            B, C, H, W = logit.size()
            if class_weight is None:
                class_weight =[1]*C #[1/C]*C

            logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob = F.softmax(logit,1)
            select = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)

        # 各个类别的损失对应的权重
        class_weight = torch.FloatTensor(class_weight).cuda().view(-1,1)
        class_weight = torch.gather(class_weight, 0, target)

        # 样本属于真实类别的概率
        prob = (prob * select).sum(1).view(-1, 1)
        prob = torch.clamp(prob, 1e-8, 1-1e-8)

        focus = torch.pow((1-prob), self.gamma)
        #focus = torch.where(focus < 2.0, focus, torch.zeros(prob.size()).cuda())
        focus = torch.clamp(focus, 0, 2)

        batch_loss = - class_weight * focus * prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss


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
    def __init__(self, loss_funs, loss_weights=None, sigmoid_flag=False):
        """
        Args:
            loss_funs: list，需要使用的损失函数
            loss_weights: list，每一个损失对应的权重
        """
        super(GetLoss, self).__init__()
        self.loss_funs = torch.nn.ModuleList(loss_funs)
        self.loss_weights = loss_weights
        self.sigmoid_flag = sigmoid_flag

        # 打印当前使用的损失函数名称
        for index, loss_fun in enumerate(loss_funs):
            print("Loss function {}: {}".format(index, type(loss_fun).__name__))

        if self.loss_weights:
            self.loss_weights = torch.Tensor(self.loss_weights)
    
    def forward(self, input, target):
        if self.sigmoid_flag:
            input = torch.sigmoid(input)
        
        loss = 0
        if self.loss_weights:
            if self.loss_weights.type() != input.data.type():
                self.loss_weights.type_as(input.data)

            for index, loss_fun in enumerate(self.loss_funs):
                loss += self.loss_weights[index] * loss_fun(input, target)
        else:
            for index, loss_fun in enumerate(self.loss_funs):
                loss += loss_fun(input, target)            

        return loss    
    

if __name__ == "__main__":
    # start_time = time.time()
    # maxe = 0
    # for i in range(1000):
    #     x = torch.rand(12800,2)*random.randint(1,10)
    #     x = Variable(x.cuda())
    #     l = torch.rand(12800).ge(0.1).long()
    #     l = Variable(l.cuda())

    #     output0 = MultiFocalLoss(gamma=0)(x,l)
    #     output1 = nn.CrossEntropyLoss()(x,l)
    #     a = output0.item()
    #     b = output1.item()
    #     if abs(a-b)>maxe: maxe = abs(a-b)
    # print('time:',time.time()-start_time,'max_error:',maxe)

    # start_time = time.time()
    # maxe = 0
    # for i in range(100):
    #     x = torch.rand(128,1000,8,4)*random.randint(1,10)
    #     x = Variable(x.cuda())
    #     l = torch.rand(128,8,4)*1000    # 1000 is classes_num
    #     l = l.long()
    #     l = Variable(l.cuda())

    #     output0 = MultiFocalLoss(gamma=0)(x,l)
    #     output1 = nn.NLLLoss2d()(F.log_softmax(x),l)
    #     a = output0.item()
    #     b = output1.item()
    #     if abs(a-b) > maxe : maxe = abs(a-b)
    # print('time:', time.time()-start_time,'max_error:',maxe)

    target0 = torch.ones(1, 2, 2)
    target0[0, 1, 1] = 0
    target1 = torch.zeros(1, 2, 2)
    target1[0, 1, 1] = 1
    target = torch.cat((target0, target1), dim=0)
    oh = one_hot(target, 2)
    print(oh)