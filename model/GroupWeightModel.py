import  torch.nn as nn
import torch
import math

class GroupWeightModel(nn.Module):
    def __init__(self, n_groups):
        super(GroupWeightModel, self).__init__()
        self.n_groups = n_groups
        self.pesu_group_weight = nn.Parameter(torch.randn(1, self.n_groups), requires_grad=True)
        self.init_weight()

    def init_weight(self):
        stdv = 1. / math.sqrt(self.pesu_group_weight.size(1))
        self.pesu_group_weight.data.uniform_(-stdv, stdv)

    def forward(self, item_loss, iw):
        '''
        
        Args:
            item_loss: shape is [batchsize * 3, ].
            e.g [item1_weak1_loss,
                item1_weak2_loss,
                item1_weak3_loss,
                ....
            ]
            iw:

        Returns:

        '''
        group_weight = torch.sigmoid(self.pesu_group_weight)
        final_weight = torch.matmul(iw.view(-1, 1), group_weight)
        
        return (final_weight * (item_loss.view(final_weight.shape))).sum()

