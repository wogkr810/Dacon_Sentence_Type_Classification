import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from utils.heads import ConvSDSHead, RobertaClassificationHead, Concat_Hidden_States
from utils.pooling import MeanMaxPooling, LSTMPooling, WeightedLayerPooling

class BaseModel(nn.Module):
    def __init__(self, input_dim=9351):
        super(BaseModel, self).__init__()
        self.feature_extract = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
        )
        self.type_classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=512, out_features=4),
        )
        self.polarity_classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=512, out_features=3),
        )
        self.tense_classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=512, out_features=3),
        )
        self.certainty_classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=512, out_features=2),
        )
            
    def forward(self, x):
        x = self.feature_extract(x)
        # 문장 유형, 극성, 시제, 확실성을 각각 분류
        type_output = self.type_classifier(x)
        polarity_output = self.polarity_classifier(x)
        tense_output = self.tense_classifier(x)
        certainty_output = self.certainty_classifier(x)
        return type_output, polarity_output, tense_output, certainty_output


# return dict를 안하니, sequce_output이랑 pooled_output이나오는거군
class RobertaLinear(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.roberta = RobertaModel(config)
        self.sds_classifier = ConvSDSHead(config)
        self.model_name = config.model_name
        self.hidden_size = config.hidden_size

        # use roberta_linear
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)

        # use roberta_class
        self.l1 = nn.Linear(self.hidden_size, 64)
        self.bn1 = nn.LayerNorm(64)
        self.l2 = nn.Linear(64, 10)

        # use roberta_dacon
        self.dropout2 = torch.nn.Dropout(0.5)
        self.linear2 = torch.nn.Sequential(
            nn.Linear(in_features = self.hidden_size, out_features = 512),
            nn.ReLU()
        )

        if self.model_name  == "roberta_class":
            in_features_size = 10
        elif self.model_name  == "roberta_dacon":
            in_features_size = 512
        else:
            in_features_size = self.hidden_size

        # classifier
        self.type_classifier = nn.Linear(in_features=in_features_size, out_features=4)
        self.polarity_classifier = nn.Linear(in_features=in_features_size, out_features=3)
        self.tense_classifier = nn.Linear(in_features=in_features_size, out_features=3)
        self.certainty_classifier = nn.Linear(in_features=in_features_size, out_features=2)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_ids, attention_mask):
        # 공식 도큐먼트보면 
        # (sequence_output, pooled_output) + encoder_outputs[1:] <- 이게 리턴값이지만,
        # encoder_outputs[0] 은 hidden_states고 1: 부터는 default에 의해서 전부 None
        # 추가적으로, None값에 의해 [1:] 부분이 생략되진않겠지만, return_dict = False -> None아닌것만 출력해서 가능함
        # sequence : hidden state -> [batch, max_length, hidden_size]
        # self. pooler -> [batch , hidden_size] -> dense로 차원 유지 ->  tanh 적용 

        # class RobertaPooler(nn.Module):
        #     def __init__(self, config):
        #         super().__init__()
        #         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        #         self.activation = nn.Tanh()

        #     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        #         # We "pool" the model by simply taking the hidden state corresponding
        #         # to the first token.
        #         first_token_tensor = hidden_states[:, 0]
        #         pooled_output = self.dense(first_token_tensor)
        #         pooled_output = self.activation(pooled_output)
        #         return pooled_output
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        sequence_output, pooled_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask,return_dict=False)
        
        x = pooled_output

        if self.model_name == "roberta_class":
            x = self.dropout(x)
            x = self.l1(x)
            x = self.bn1(x)
            x = torch.nn.Tanh()(x)
            x = self.dropout(x)
            x = self.l2(x)
            x = self.dropout(x)

        elif self.model_name == "roberta_linear":
            x = self.dropout(x)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)

        elif self.model_name == "roberta_dacon":
            x = self.dropout2(x)
            x = self.linear2(x)

        elif self.model_name == "roberta_sds":
            x = self.sds_classifier(sequence_output)
            type_output = self.type_classifier(x)[:,0,:]
            polarity_output = self.polarity_classifier(x)[:,0,:]
            tense_output = self.tense_classifier(x)[:,0,:]
            certainty_output = self.certainty_classifier(x)[:,0,:]
            return type_output, polarity_output, tense_output, certainty_output


        type_output = self.type_classifier(x)
        polarity_output = self.polarity_classifier(x)
        tense_output = self.tense_classifier(x)
        certainty_output = self.certainty_classifier(x)

        return type_output, polarity_output, tense_output, certainty_output

# last_hidden_state를 쓰는거구나..
class RobertaDocument(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)  # add_pooling_layer하면 return dict True했을 때, pooled output 안나옴
        self.classifier = RobertaClassificationHead(config)
        self.sds_classifier = ConvSDSHead(config)
        self.concat_hidden_classifier = Concat_Hidden_States(config)
        self.mean_max_pooling = MeanMaxPooling(config)
        
        self.model_name = config.model_name
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers

        self.hiddendim_lstm = 256
        self.lstm_pooler = LSTMPooling(self.num_hidden_layers, self.hidden_size,self.hiddendim_lstm, config)

        self.layer_start = 22
        self.weighted_pooler = WeightedLayerPooling(config, num_hidden_layers = self.num_hidden_layers, layer_start = self.layer_start, layer_weights = None)

        self.softmax = nn.Softmax(dim=1)

        if self.model_name  == "roberta_document_concat_hidden":
            in_features_size = self.hidden_size * 4
        elif self.model_name == "roberta_document_mean_max":
            in_features_size = self.hidden_size * 2
        else:
            in_features_size = self.hidden_size

        self.type_classifier = nn.Sequential(
            nn.Linear(in_features = in_features_size, out_features=4),
        )
        self.polarity_classifier = nn.Sequential(
            nn.Linear(in_features = in_features_size, out_features=3),
        )
        self.tense_classifier = nn.Sequential(
            nn.Linear(in_features = in_features_size, out_features=3),
        )
        self.certainty_classifier = nn.Sequential(
            nn.Linear(in_features = in_features_size, out_features=2),
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.model_name == "roberta_document_weighted":
            return_dict = False

        output_hidden_states = (
                True
                if self.model_name in ["roberta_document_concat_hidden", "roberta_document_lstm", "roberta_document_weighted"] else output_hidden_states
            )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0] # outputs[0] == outputs.last_hidden_state == sequence_output 랑 같음 

        if self.model_name == "roberta_document_linear":
            logits = self.classifier(sequence_output)

        elif self.model_name == "roberta_document_concat_hidden":
            logits = self.concat_hidden_classifier(outputs.hidden_states)
        
        elif self.model_name == "roberta_document_mean_max":
            logits = self.mean_max_pooling(sequence_output)

        elif self.model_name == "roberta_document_lstm":
            logits = self.lstm_pooler(outputs.hidden_states)

        elif self.model_name == "roberta_document_weighted":
            x = torch.stack(outputs[2])
            logits = self.weighted_pooler(x)[:, 0]

        elif self.model_name == "roberta_document_sds":
            logits = self.sds_classifier(sequence_output)
            type_output = self.type_classifier(logits)[:,0,:]
            polarity_output = self.polarity_classifier(logits)[:,0,:]
            tense_output = self.tense_classifier(logits)[:,0,:]
            certainty_output = self.certainty_classifier(logits)[:,0,:]

            return type_output, polarity_output, tense_output, certainty_output

        type_output = self.type_classifier(logits)
        polarity_output = self.polarity_classifier(logits)
        tense_output = self.tense_classifier(logits)
        certainty_output = self.certainty_classifier(logits)

        # type_output = self.softmax(self.type_classifier(logits))
        # polarity_output = self.softmax(self.polarity_classifier(logits))
        # tense_output = self.softmax(self.tense_classifier(logits))
        # certainty_output = self.softmax(self.certainty_classifier(logits))
        
        return type_output, polarity_output, tense_output, certainty_output

        # loss = None
        # if labels is not None:
        #     if self.config.problem_type is None:
        #         if self.num_labels == 1:
        #             self.config.problem_type = "regression"
        #         elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
        #             self.config.problem_type = "single_label_classification"
        #         else:
        #             self.config.problem_type = "multi_label_classification"

        #     if self.config.problem_type == "regression":
        #         loss_fct = MSELoss()
        #         if self.num_labels == 1:
        #             loss = loss_fct(logits.squeeze(), labels.squeeze())
        #         else:
        #             loss = loss_fct(logits, labels)
        #     elif self.config.problem_type == "single_label_classification":
        #         loss_fct = CrossEntropyLoss()
        #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        #     elif self.config.problem_type == "multi_label_classification":
        #         loss_fct = BCEWithLogitsLoss()
        #         loss = loss_fct(logits, labels)


        # if not return_dict:
        #     output = (logits,) + outputs[2:]
        #     return ((loss,) + output) if loss is not None else output

        # return SequenceClassifierOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )


