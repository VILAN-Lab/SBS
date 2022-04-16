import json
import os
import pickle
import time
from os.path import join

import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import random
import copy


def compute_score_with_logits(logits, labels):
    logits = torch.argmax(logits, 1)
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def train_bias(model, train_loader, eval_loader,args,qid2type):
    num_epochs=args.bias_epochs
    output=args.output
    loss_weight = args.loss_weight
    optim = torch.optim.Adamax(model.parameters())
    total_step = 0

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0

        t = time.time()
        for i, (v, q, a, b, hintscore,type_mask,notype_mask,q_mask) in tqdm(enumerate(train_loader), ncols=100,
                                                   desc="Epoch %d" % (epoch + 1), total=len(train_loader)):

            total_step += 1
            #########################################
            v = Variable(v).cuda().requires_grad_()
            q = Variable(q).cuda()
            a = Variable(a).cuda()
            b = Variable(b).cuda()
            #########################################

            pred, loss,_ = model(v, q, a, b, None, loss_weight)
            if (loss != loss).any():
                raise ValueError("NaN loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            total_loss += loss.item() * q.size(0)
            batch_score = compute_score_with_logits(pred, a.data).sum()
            train_score += batch_score

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        print("\tEpochs %d" %(epoch+1), "loss='%.2f'" % total_loss, "train_score='%.2f'" % train_score)

    model_path = os.path.join(output, 'bias_model.pth')
    print('Save model at', model_path)
    torch.save(model, model_path)
