import sys
import os
from transformers import optimization

rootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootPath)

import torch
from torch import nn
import pandas as pd
from data_process import *
from similarity import *


# 训练模型
def train():
    model = albert_similarity_model()

    train, test_token,test_mask,test_seg_ment, eval_y = load_data()

    eval_token = torch.tensor(test_token)
    eval_mask = torch.tensor(test_mask)
    eval_seg_ment = torch.tensor(test_seg_ment)
    if torch.cuda.is_available():
        model = model.cuda()
        eval_token = eval_token.cuda()
        eval_mask = eval_mask.cuda()
        eval_seg_ment = eval_seg_ment.cuda()

    optimizer = optimization.AdamW(model.parameters(), lr=1e-3)
    loss_func = nn.BCELoss()

    best_acc = 0

    for epoch in range(8):
        for step, ((token,mask,seg_ment),label) in enumerate(train):
            if torch.cuda.is_available():
                token = torch.tensor(token).cuda()
                mask = torch.tensor(mask).cuda()
                seg_ment = torch.tensor(seg_ment).cuda()
                label = torch.tensor(label).cuda()
            output = model(token,mask,seg_ment)
            loss = loss_func(output, label.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 20 == 0:
                test_output = model(eval_token,eval_mask,eval_seg_ment)
                pred_y = (test_output.cpu().data.numpy() > 0.5).astype(int)
                accuracy = float((pred_y == eval_y).astype(int).sum()) / float(eval_y.size)
                if accuracy > best_acc:
                    best_acc = accuracy
                    torch.save(model.state_dict(), 'model_s.pth')
                    print('save model, accuracy: %.3f' % accuracy)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(),
                      '| test accuracy: %.3f' % accuracy)


if __name__ == '__main__':
    train()