import json
import os
import pickle
import time
from os.path import join

import torch
import torch.nn as nn
import torch.optim as optim
import utils
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm

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


def repair_data(model, train_loader, num_epochs, answer_voc_size):

    print("build labels mat")
    labels = torch.tensor([torch.argmax(data[2]) for data
                           in train_loader.dataset]).long().cuda()
    print("labels mat size=", labels.size())
    n_cls = answer_voc_size
    print("n_cls=", n_cls)
    print("build cls_idx mat")
    cls_idx = torch.stack([labels == c for c in range(n_cls)]).float().cuda()
    print("cls_idx mat size=", cls_idx.size())

    weight_param = nn.Parameter((torch.ones(len(train_loader.dataset))/2).cuda())
    print("weight_param mat size=", weight_param.size())
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer_w = optim.SGD([weight_param], lr=10)

    total_step = 0

    for epoch in range(num_epochs):
        train_score = 0
        losses = []
        t = time.time()

        for i, (v, q, a, b, _, idx) in tqdm(enumerate(train_loader), ncols=100,
                    desc="Epoch %d" % (epoch+1), total=len(train_loader)):

            total_step += 1
            v = Variable(v).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()
            b = Variable(b).cuda()

            # class probabilities
            w = torch.sigmoid(weight_param)
            z = (w[idx] / w.mean()).view(v.size(0), 1)
            cls_w = torch.matmul(cls_idx, w)
            c_w = cls_w / cls_w.sum()

            # classifier
            pred = model(v, None, q, a, b)
            loss = instance_bce_with_logits(pred, a)
            loss = (z * loss).mean()
            loss = loss * v.size(0)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # class weights
            optimizer_w.zero_grad()
            # get the idx of the right answer
            a_arg = torch.argmax(a, dim=1)
            entropy = -(c_w[a_arg].log() * z).mean()
            loss_w = 1 - loss / entropy
            loss_w.backward()
            optimizer_w.step()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            train_score += batch_score

        total_loss = sum(losses) / len(losses)
        train_score = 100 * train_score / len(train_loader.dataset)
        print("loss='%.2f'" % total_loss, "train_score='%.2f'" % train_score)

    # class probabilities & bias
    with torch.no_grad():
        w = torch.sigmoid(weight_param)
        cls_w = torch.matmul(cls_idx, w)
        h = cls_w / cls_w.sum()
        rnd_loss = -(h * (h + 1e-7).log()).sum().item()
        bias = 1 - loss / rnd_loss
    print('train_score = {:.2f}%, Loss = {:.3f}, Rndloss = {:.3f}, Bias = {:.3f}'
          .format(train_score, loss, rnd_loss, bias))

    return w, h, cls_idx, cls_w, bias
