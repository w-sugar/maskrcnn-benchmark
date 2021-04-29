import torch
from torch.nn import functional as F
from torch import nn

class CrossEntropyLossSmooth(nn.Module):
    def __init__(self):
        super(CrossEntropyLossSmooth, self).__init__()

    def forward(self, fc_out, label, target_value, weight=None):
        target_value = target_value.unsqueeze(1)
        one_hot_lable = torch.FloatTensor(fc_out.shape[0], fc_out.shape[1]).to(device=label.device)
        one_hot_lable.zero_()
        one_hot_lable.scatter_(1, torch.reshape(label, (fc_out.shape[0], 1)), target_value)
        loss = one_hot_lable * torch.softmax(fc_out, 1)
        if weight is None:
            loss = -torch.sum(torch.log(torch.sum(loss, 1)))/fc_out.shape[0] + 1e-4
        else:
            weight_ce = torch.gather(weight, 0, label)
            loss_index = torch.log(torch.sum(loss, 1))
            loss = -torch.sum(weight_ce * loss_index) / torch.sum(weight_ce)
        return loss

if __name__ == '__main__':
    lable = torch.tensor([1,0,1,2,1])
    target_value = torch.tensor([1, 1, 1,1,1], dtype=torch.float32)

    fc_out = torch.tensor(
    [
        [2.5, -2, 0.8989],
        [3, 0.8, -865],
        [0.00000000000001, 2, 4.9],
        [11,23,55],
        [44,1,5]
    ])
    weight_ce = torch.tensor([100, 22, 10]).to(dtype=torch.float32)
    loss1 = CrossEntropyLossSmooth()
    print(F.cross_entropy(fc_out, lable, weight_ce))
    print(loss1(fc_out, lable, target_value, weight_ce))