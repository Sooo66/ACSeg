import torch.nn.functional as F
import torch

# class ModularityLoss(torch.nn.Module):
#     def __init__(self):
#         super(ModularityLoss, self).__init__()

#     def forward(self, x, delta, m):
#         loss = torch.sum(delta * x, dim=(1, 2))
#         loss = loss * (-1 / (2 * m))

#         return loss.sum()


def ModularityLoss(x, delta, m):
    loss = torch.sum(delta * x, dim=(1, 2))
    loss = loss * (-1 / m)

    return loss.sum()
