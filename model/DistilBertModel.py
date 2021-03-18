from transformers import DistilBertPreTrainedModel, DistilBertModel
import torch.nn as nn
import torch
class DistilBertForSequenceClassification(DistilBertPreTrainedModel):
    def __init__(self, config):
        super(DistilBertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.config = config
        self.c_count = 1
        self.init_weights()

    def expand_class_head(self, c_count):
        self.c_count = c_count
        if c_count > 1:
            for i in range(1, c_count+1):
                setattr(self, "classifier_{}".format(i), nn.Linear(self.config.dim, self.config.num_labels))

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, labels=None,
                reduction="mean", is_gold=True):
        distilbert_output = self.distilbert(
            input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        if self.c_count == 1 or is_gold:
            logits = self.classifier(pooled_output)  # (bs, dim)
        else:
            logits = []
            for i in range(1, self.c_count+1):
                logits.append(self.__dict__['_modules']["classifier_{}".format(i)](pooled_output))
            logits = torch.cat(logits, dim=1)

        outputs = (logits,) + distilbert_output[1:]
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss(reduction=reduction)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)