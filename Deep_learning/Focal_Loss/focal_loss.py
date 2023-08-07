import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# TODO Test the loss
if __name__ == '__main__':
    loss_function_mean = FocalLoss()
    loss_function_sum = FocalLoss(reduction='sum')
    a = torch.tensor([0., 1, 0, 0, 0])# target
    b = torch.tensor([1., 1, 1, 1, 1])
    focal_loss_mean = loss_function_mean(b, a)
    focal_loss_sum = loss_function_sum(b, a)
    print(f'focal loss mean is {focal_loss_mean}')
    print(f'focal loss sum is {focal_loss_sum}')
    ce_loss_function = nn.CrossEntropyLoss()
    ce_loss = ce_loss_function(b, a)
    print(f'ce loss is {ce_loss}')