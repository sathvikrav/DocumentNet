import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def pairwise_loss_updated(outputs1, outputs2, similarity):
    dot_product = torch.mm(outputs1, outputs2.t())

    mask_positive = similarity.data > 0
    mask_negative = similarity.data <= 0
    exp_loss = torch.log(1+torch.exp(-torch.abs(dot_product))) + torch.max(dot_product, Variable(torch.FloatTensor([0.])))-similarity * dot_product
    
    #weight
    S1 = torch.sum(mask_positive.float())
    S0 = torch.sum(mask_negative.float())
    S = S0+S1
    
    exp_loss[similarity.data > 0] = exp_loss[similarity.data > 0] * (S / S1)
    exp_loss[similarity.data <= 0] = exp_loss[similarity.data <= 0] * (S / S0)

    loss = torch.sum(exp_loss) / S

    return loss