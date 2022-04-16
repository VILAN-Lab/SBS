from __future__ import division
import json
import os
import pickle
import time
from os.path import join

import torch
import torch.nn as nn
import torch.optim as optim
from language_model import WordEmbedding
from dataset import Dictionary
import utils
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import random

def tokenize(dictionary, ans, max_length=3):
    tokens = dictionary.tokenize(ans, False)
    tokens = tokens[:max_length]
    if len(tokens) < max_length:
        padding = [dictionary.padding_idx] * (max_length - len(tokens))
        tokens = padding + tokens
    return tokens


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def select(model, dataloader, qid2type):
    idx_keep = []
    idx_abandon = []
    total_yesno = 0
    total_other = 0
    total_number = 0

    for v, q, a, b, qids, idx in tqdm(dataloader, ncols=100,
                                    total=len(dataloader), desc="select"):
        v = Variable(v.cuda())
        q = Variable(q.cuda())
        a = Variable(a.cuda())
        batch_size = v.size(0)

        pred = model(v, None, q, None, None)  # [B, 2274]
        pred_idx = torch.max(pred, 1)[1].data  # [B, 1] 1 represent idx
        gt_idx = torch.argmax(a, 1)  # [B, 1]

        qids = qids.detach().cpu().int().numpy()

        # for j in range(len(qids)):
        for i in range(batch_size):
            # random_num = random.randint(1,2)
            # if random_num == 1:
            #     idx_abandon.append(idx[i].item())
            # else:
            #     idx_keep.append(idx[i].item())

            if pred_idx[i].item() == gt_idx[i].item():
                idx_abandon.append(idx[i].item())
            else:
                idx_keep.append(idx[i].item())


    select_ratio = len(idx_keep) / (len(idx_keep) + len(idx_abandon))
    yn_ratio = total_yesno / (len(idx_keep) + len(idx_abandon))
    num_ratio = total_number / (len(idx_keep) + len(idx_abandon))
    other_ratio = total_other / (len(idx_keep) + len(idx_abandon))

    print('idx_keep: ', len(idx_keep))
    print('total idx: ', len(idx_keep) + len(idx_abandon))
    print('total keep / total is %.2f' % select_ratio)
    print('yn keep / total is %.2f' % yn_ratio)
    print('num keep / total is %.2f' % num_ratio)
    print('other keep / total is %.2f' % other_ratio)

    with open('list_keep.json', 'w') as list_keep:
        json.dump(idx_keep, list_keep)

    with open('list_abandon.json', 'w') as list_abandon:
        json.dump(idx_abandon, list_abandon)

    return select_ratio





