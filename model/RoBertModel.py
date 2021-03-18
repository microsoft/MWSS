from transformers import BertPreTrainedModel, RobertaConfig, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP, RobertaModel
import torch
import torch.nn as nn

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 1, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
class RobertaForSequenceClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForSequenceClassification, self).__init__(config)
        self.num_labels = 2

        self.roberta = RobertaModel(config)
        # self.classifier = RobertaClassificationHead(config)
        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
        self.c_count = 1
        
    def expand_class_head(self, c_count):
        self.c_count = c_count
        if c_count > 1:
            for i in range(1, c_count+1):
                setattr(self, "classifier_{}".format(i), nn.Linear(self.config.hidden_size, self.num_labels))
                
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        reduction="mean", is_gold=True
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # sequence_output = outputs[0]
        # logits = self.classifier(sequence_output)
        hidden_state = outputs[0]
        pooled_output = hidden_state[:, 1]  # (bs, dim)
        outputs = (pooled_output.detach(),)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)

        if self.c_count == 1 or is_gold:
            logits = self.classifier(pooled_output)  # (N, C)
            # logits = self.classifier(sequence_output)  # (N, C)
        else:
            logits = []
            for i in range(1, self.c_count+1):
                logits.append(self.__dict__['_modules']["classifier_{}".format(i)](pooled_output))
            logits = torch.cat(logits, dim=1)

        outputs = (logits, ) + outputs
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction=reduction)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits