import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import warnings
from transformers import DistilBertModel
class CNN_Text(nn.Module):

    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args

        # V = args.embed_num
        # D = args.embed_dim
        C = args.num_labels
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        # self.embed = nn.Embedding(V, D)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]

        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(args.dropout)
        self.classifier = nn.Linear(len(Ks) * Co, C)
        self.Ks = Ks
        self.Co = Co
        self.C = C
        self.c_count = 1
        self.Ci= Ci
        self.embed_shape = [30522, 768]
        self.embed = nn.Embedding(self.embed_shape[0], self.embed_shape[1])
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(self.Ci, self.Co, (K, self.embed_shape[1]))
             for K in self.Ks])
        # self.from_pretrained("distilbert-base-uncased")

    def from_pretrained(self, model_name_or_path):
        distilbert = DistilBertModel.from_pretrained(model_name_or_path)
        state_dict = distilbert.state_dict()
        embed_weight = state_dict['embeddings.word_embeddings.weight']
        self.embed.from_pretrained(embed_weight)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def expand_class_head(self, c_count):
        self.c_count = c_count
        if c_count > 1:
            for i in range(1, c_count + 1):
                setattr(self, "classifier_{}".format(i), nn.Linear(len(self.Ks) * self.Co, self.C))

    def forward(self, input_ids, attention_mask=None, labels=None, reduction="mean", is_gold=True):
        input_ids = self.embed(input_ids)  # (N, W, D)

        # if self.args.static:
        #     input_ids = Variable(input_ids)

        input_ids = input_ids.unsqueeze(1)  # (N, Ci, W, D)

        input_ids = [F.relu(conv(input_ids)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        input_ids = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in input_ids]  # [(N, Co), ...]*len(Ks)

        hidden_state = torch.cat(input_ids, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''

        outputs = (hidden_state.detach(), )
        hidden_state = self.dropout(hidden_state)  # (N, len(Ks)*Co)

        if self.c_count == 1 or is_gold:
            logits = self.classifier(hidden_state)  # (N, C)
        else:
            logits = []
            for i in range(1, self.c_count+1):
                logits.append(self.__dict__['_modules']["classifier_{}".format(i)](hidden_state))
            logits = torch.cat(logits, dim=1)

        outputs = (logits, ) + outputs
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction=reduction)
            loss = loss_fct(logits.view(-1, self.C), labels.view(-1))
            outputs = (loss,) + outputs


        return outputs
