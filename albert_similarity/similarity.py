from transformers import AlbertModel
import torch
from torch import nn
from torch.functional import  F

class albert_similarity_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.albert = AlbertModel.from_pretrained('clue/albert_chinese_tiny')
        for param in self.albert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(312, 1)

    def forward(self,x,mask,seg_ment):
        _, pooled = self.albert(input_ids=x,attention_mask=mask,token_type_ids=seg_ment)
        out = self.fc(pooled)
        out = out.squeeze(-1)
        out = F.sigmoid(out)
        return out






