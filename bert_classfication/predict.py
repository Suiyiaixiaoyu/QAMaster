import torch
from bert_classfication.data_process import *
from bert_classfication.classification import *
import os
from transformers import BertTokenizer,BertModel,BertForMaskedLM,AlbertModel
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import numpy as np
from torch.functional import  F

class bertclassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.albert = AlbertModel.from_pretrained('clue/albert_chinese_tiny')
        for param in self.albert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(312, 1)

    def forward(self,x,mask):
        _, pooled = self.albert(x,attention_mask=mask,return_dict=False)
        out = self.fc(pooled)
        out = out.squeeze(-1)
        out = F.sigmoid(out)
        return out
path = os.path.dirname(__file__)

model = bertclassification()
pre_dict = model.state_dict()
model_dict = torch.load(path + '/model.pth')
pre_dict.update(model_dict)
model.load_state_dict(pre_dict)
model = model.cuda()
model.eval()


def classification_predict(s):
    token,mask = seq2index(s)
    token = torch.tensor(token).cuda()
    mask = torch.tensor(mask).cuda()
    out = model(token,mask)
    return out.cpu().data.numpy()


if __name__ == '__main__':
    while 1:
        s = input('句子：')
        print(classification_predict([s]))
