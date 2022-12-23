'''
https://github.com/Aroma-Jewel/KERC-2022-4th/blob/main/utils/heads.py
참고

'''


import torch
import string
import collections
import numpy as np
import torch.nn as nn

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.activation = nn.Tanh()
        # self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        # x = self.dropout(x)
        # x = self.dense(x)
        # x = torch.tanh(x)
        # x = self.dropout(x)
        # # x = self.out_proj(x)
        return x

# SDS Conv
class ConvSDSLayer(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_size, out_channels=input_size * 2, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(in_channels=input_size * 2, out_channels=input_size, kernel_size=1,)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = x + self.activation(out)
        out = self.layer_norm(out)
        return out


class ConvSDSHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        convs = []
        for n in range(5):
            convs.append(ConvSDSLayer(len(x[0]), self.config.hidden_size).cuda())
        self.convs = nn.Sequential(*convs)
        out = self.convs(x)
        return out

# class ConvSDSHead(nn.Module):
#     def __init__(
#         self, config, num_labels: int = 3
#     ):
#         super().__init__()
#         self.config = config
#         self.classifier = nn.Linear(config.hidden_size, num_labels).cuda()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         convs = []
#         for n in range(5):
#             convs.append(ConvSDSLayer(len(x[0]), self.config.hidden_size).cuda())
#         self.convs = nn.Sequential(*convs)
#         out = self.convs(x)
#         return self.classifier(out)[:,0,:]


class Hidden_States_Outputs(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dense = nn.Linear(config.hidden_size*4, config.hidden_size*4).cuda()
    classifier_dropout = (
      config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
    )

    self.dropout = nn.Dropout(classifier_dropout)
    # self.out_proj = nn.Linear(config.hidden_size*4, num_labels).cuda()
    self.tanh = nn.Tanh()
    
  def forward(self, x):
    x = self.dropout(x)
    x = self.dense(x)
    x = self.tanh(x)
    x = self.dropout(x)
    # x = self.out_proj(x)
    return x


class Concat_Hidden_States(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.hidden_concat_outputs = Hidden_States_Outputs(config)
    # self.out_proj = nn.Linear(config.hidden_size*4, config.num_labels).cuda()
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # stacked_output = torch.cat(all_hidden_states[-4:],-1) 과 같음
    # https://stackoverflow.com/questions/70682546/extract-and-concanate-the-last-4-hidden-states-from-bert-model-for-each-input 
    # 참고
    all_hidden_states = torch.stack(x)
    
    stacked_output = torch.cat((all_hidden_states[-4],all_hidden_states[-3], all_hidden_states[-2], all_hidden_states[-1]),-1)
    sequence_stacked_output = stacked_output[:,0]
    sequence_stacked_output = self.hidden_concat_outputs(sequence_stacked_output)
    
    # logits = self.out_proj(sequence_stacked_output)
    return sequence_stacked_output