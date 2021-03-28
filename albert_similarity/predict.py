import torch
import os
from albert_similarity.data_process import *

path = os.path.dirname(__file__)
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
        _, pooled = self.albert(input_ids=x,attention_mask=mask,token_type_ids=seg_ment,return_dict=False)
        out = self.fc(pooled)
        out = out.squeeze(-1)
        out = F.sigmoid(out)
        return out

model = albert_similarity_model()
pre_dict = model.state_dict()
model_dict = torch.load(path + '/model_s.pth')
pre_dict.update(model_dict)
model.load_state_dict(pre_dict)
model = model.cuda()
model.eval()


def predict(p, q):
    token,mask,segment = seq2index(p,q)
    token = torch.tensor(token).cuda()
    mask = torch.tensor(mask).cuda()
    segment = torch.tensor(segment).cuda()
    out = model(token,mask,segment)
    return out.cpu().data.numpy()


if __name__ == '__main__':
    print(predict(['北京烤鸭在哪？'], ['关于北京烤鸭']))
