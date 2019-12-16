import torch.nn as nn
import torch.nn.functional as F
import torch


def nll_loss(output, target):
    output = output.view(-1, output.size(-1))
    target = target.view(-1)
    return nn.NLLLoss()(output, target)

def maskNLLLoss(output, target, mask=None):
    #nTotal = mask.sum()
    # print(output.size())
    # print(target.size())
    output = output.view(-1, output.size(-1))
    target = target.view(-1)
    # print(output, target)
    selected = torch.gather(output, 1, target.view(-1, 1))
    # print(selected)
    #crossEntropy = -torch.log(torch.gather(output, 1, target.view(-1, 1)).squeeze(1))
    crossEntropy = -selected
    # loss = crossEntropy.masked_select(mask).mean()
    # print(crossEntropy)
    loss = crossEntropy.mean()
    # print(loss)
    # loss = loss.to(device)
    #return loss, nTotal.item()
    return loss

def bce_loss(output, target):
    return nn.BCELoss()(output.double(), target.double())
