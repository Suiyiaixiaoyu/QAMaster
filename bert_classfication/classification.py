from transformers import BertTokenizer,BertModel,BertForMaskedLM,AlbertModel
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn
import jieba
import pandas as pd
from torch.functional import  F

class bertclassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.albert = AlbertModel.from_pretrained('clue/albert_chinese_tiny')
        for param in self.albert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(312, 1)

    def forward(self,x,mask):
        _, pooled = self.albert(x,attention_mask=mask)
        out = self.fc(pooled)
        out = out.squeeze(-1)
        out = F.sigmoid(out)
        return out




