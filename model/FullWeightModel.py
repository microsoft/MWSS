import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import DistilBertModel

class FullWeightModel(nn.Module):
    def __init__(self, n_groups, hidden_size):
        super(FullWeightModel, self).__init__()
        self.n_groups = 1 if n_groups == 0 else n_groups
        #self.pesu_group_weight = nn.Parameter(torch.randn(1, self.n_groups), requires_grad=True)
        self.embed_shape = [30522, 768]
        # self.ins_embed = nn.Embedding(self.embed_shape[0], self.embed_shape[1])
        self.cls_emb = 256 #self.embed_shape[1]
        h_dim = 768
        self.y_embed = nn.Embedding(2, self.cls_emb) 

        #self.ins_weight = nn.Linear(hidden_size+self.cls_emb, 1)

        self.ins_weight = nn.Sequential(
            nn.Linear(hidden_size+self.cls_emb, h_dim),
            nn.ReLU(), #Tanh(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(), #Tanh(),
            nn.Linear(h_dim, 1)
        )
        
    def reset_groups(self, new_groups):
        self.n_groups = new_groups

    def forward(self, x_feature, y_weak, item_loss=None):
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
        # detach the feature
        x_feature = x_feature.detach()
        feature_dim = x_feature.shape[-1]
        x_feature = x_feature.repeat(1, self.n_groups).view(-1, feature_dim)
        y_emb = self.y_embed(y_weak).view(-1, self.cls_emb)
        #ATTENTION: weight depends on the pair of feature and weak label instead of the source.
        #final_weight = F.softmax(self.ins_weight(torch.cat([x_feature, y_emb], dim=-1)), dim=0).squeeze()
        #return (final_weight * item_loss).sum() ,final_weight

        # sigmoid with mean
        final_weight = torch.sigmoid(self.ins_weight(torch.cat([x_feature, y_emb], dim=-1))).squeeze()
        if item_loss is None:
            return final_weight
        else:
            return (final_weight * item_loss).mean(), final_weight
    
        
        

        #final_weight = F.relu(self.ins_weight(torch.cat([x_feature, y_emb], dim=-1))).squeeze()
        #return (final_weight * item_loss).mean()


