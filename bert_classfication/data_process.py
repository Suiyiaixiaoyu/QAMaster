import pandas as pd
from transformers import BertTokenizer, AlbertModel
from collections import defaultdict
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch

path = os.path.dirname(__file__)
tokenizer = BertTokenizer.from_pretrained("clue/albert_chinese_tiny")

def load_data():
    df = pd.read_csv('../data/classification.csv')
    train_df, eval_df = split_data(df)
    train_x = train_df['sentence'].values
    train_y = train_df['label'].values
    eval_x = eval_df['sentence'].values
    eval_y = eval_df['label'].values

    train_token,train_mask = seq2index(train_x)

    train_dict = {
        'token':train_token,
        'mask':train_mask,
        'label':train_y
    }
    test_token,test_mask = seq2index(eval_x)
    train_dataset = bertdataset(train_dict)
    train = DataLoader(train_dataset,batch_size=64,shuffle=True)
    return train, test_token,test_mask, eval_y
# 划分训练集和测试集

class bertdataset(Dataset):
    def __init__(self,data):
        self.data = data

    def __len__(self):
        return len(self.data['token'])

    def __getitem__(self, idx):
        return (self.data['token'][idx],self.data['mask'][idx]),self.data['label'][idx]


def split_data(df):
    df = df.sample(frac=1)
    length = len(df)
    train_data = df[0:length - 1000]
    eval_data = df[length - 1000:]

    return train_data, eval_data


# 把数据转换成index
def seq2index(sentances,max_len=20):
    seg_indexes=[]
    maskes=[]
    for seg in sentances:
        seg = tokenizer.tokenize(seg)
        token = ['[CLS]'] + seg
        seq_len = len(token)
        mask = []
        seg_index = tokenizer.convert_tokens_to_ids(token)

        if max_len:
            if seq_len < max_len:
                seg_index = seg_index+[0]*(max_len-seq_len)
                mask = [1]*seq_len+[0]*(max_len-seq_len)
            else:
                seg_index = seg_index[:max_len]
                mask = [1]*max_len
                seq_len = max_len
        seg_indexes.append(seg_index)
        maskes.append(mask)
    return np.array(seg_indexes),np.array(maskes)
