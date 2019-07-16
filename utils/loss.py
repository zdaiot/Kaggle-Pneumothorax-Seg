import numpy as np
import time
import random

import torch
from torch import nn 
import torch.nn.functional as F
from torch.autograd import Variable


class DiceLoss(nn.Module):
    """two class dice loss
    """
    def __init__(self, sigmoid_flag):
        super(DiceLoss, self).__init__()
        self.sigmoid_flag = sigmoid_flag

    def forward(self, input, target):
        Num = input.size(0)
        smooth = 1

        if self.sigmoid_flag:
            input = torch.sigmoid(input)
        
        input_flat = input.view(Num, -1)
        target_flat = target.view(Num, -1)

        intersection = input_flat * target_flat
        loss = 2 * (intersection.sum(1) + smooth) / ((input_flat.sum(1) + target_flat.sum(1)) + smooth)
        loss = loss / Num

        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
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


if __name__ == "__main__":
    start_time = time.time()
    maxe = 0
    for i in range(1000):
        x = torch.rand(12800,2)*random.randint(1,10)
        x = Variable(x.cuda())
        l = torch.rand(12800).ge(0.1).long()
        l = Variable(l.cuda())

        output0 = FocalLoss(gamma=0)(x,l)
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

        output0 = FocalLoss(gamma=0)(x,l)
        output1 = nn.NLLLoss2d()(F.log_softmax(x),l)
        a = output0.item()
        b = output1.item()
        if abs(a-b)>maxe: maxe = abs(a-b)
    print('time:',time.time()-start_time,'max_error:',maxe)