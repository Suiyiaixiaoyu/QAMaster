import pandas as pd
from transformers import BertTokenizer, AlbertModel
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch

path = os.path.dirname(__file__)
tokenizer = BertTokenizer.from_pretrained("clue/albert_chinese_tiny")

def load_data():
    df = pd.read_csv('../data/LCQMC.csv')
    train_df, eval_df = split_data(df)
    train_x1 = train_df['sentence1'].values
    train_x2 = train_df['sentence2'].values
    train_y = train_df['label'].values
    eval_x1 = eval_df['sentence1'].values
    eval_x2 = eval_df['sentence2'].values
    eval_y = eval_df['label']

    train_token,train_mask,train_seg_ment = seq2index(train_x1,train_x2)

    train_dict = {
        'token':train_token,
        'mask':train_mask,
        'seg_ment':train_seg_ment,
        'label':train_y
    }
    test_token,test_mask,test_seg_ment = seq2index(eval_x1,eval_x2)
    train_dataset = bertdataset(train_dict)
    train = DataLoader(train_dataset,batch_size=512,shuffle=True)
    return train, test_token,test_mask,test_seg_ment, eval_y
# 划分训练集和测试集

class bertdataset(Dataset):
    def __init__(self,data):
        self.data = data

    def __len__(self):
        return len(self.data['token'])

    def __getitem__(self, idx):
        return (self.data['token'][idx],self.data['mask'][idx],self.data['seg_ment'][idx]),self.data['label'][idx]


def split_data(df):
    df = df.sample(frac=1)
    length = len(df)
    train_data = df[0:length - 1000]
    eval_data = df[length - 1000:]

    return train_data, eval_data


# 把数据转换成index
def seq2index(sentence1,sentence2,max_len=30):
    seg_indexes=[]
    maskes=[]
    seg_ments = []
    for index in range(len(sentence1)):
        seg1 = tokenizer.tokenize(sentence1[index])
        seg2 = tokenizer.tokenize(sentence2[index])
        token = ['[CLS]'] + seg1 +['[SEP]'] + seg2 +['[SEP]']
        seq_len = len(token)
        mask = []
        seg_index = tokenizer.convert_tokens_to_ids(token)
        seg_ment = [0]*(len(seg1)+2)+[1]*(len(seg2)+1)

        if max_len:
            if seq_len < max_len:
                seg_index = seg_index+[0]*(max_len-seq_len)
                mask = [1]*seq_len+[0]*(max_len-seq_len)
                seg_ment = seg_ment+[0]*(max_len-seq_len)
            else:
                seg_index = seg_index[:max_len]
                mask = [1]*max_len
                seq_len = max_len
                seg_ment = seg_ment[:max_len]
        seg_indexes.append(seg_index)
        maskes.append(mask)
        seg_ments.append(seg_ment)
    return np.array(seg_indexes),np.array(maskes),np.array(seg_ments)
