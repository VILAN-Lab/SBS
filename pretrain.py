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


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def pretrain(model, train_loader, num_epochs, output, dataset, preset, ispre2=False):
    utils.create_dir(output)
    model = model.cuda()

    if ispre2 == False:
        optim = torch.optim.Adamax(model.parameters())
    else:
        optim = torch.optim.Adamax([
            {'params': model.w_emb.parameters(), 'lr': 0.0002},
            {'params': model.q_emb.parameters(), 'lr': 0.0002},
            {'params': model.q_net.parameters(), 'lr': 0.0002},
            {'params': model.classifier1.parameters(), 'lr': 0.002},
        ])
    logger = utils.Logger(os.path.join(output, 'prelog.txt'))

    total_step = 0

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0

        t = time.time()

        for i, (v, q, a, _, _) in tqdm(enumerate(train_loader), ncols=100,
                                    desc="Epoch %d" % (epoch+1), total=len(train_loader)):
            total_step += 1
            v = Variable(v).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()

            pred = model(v, None, q, None, None)
            loss = instance_bce_with_logits(pred, a)

            if (loss != loss).any():
              raise ValueError("NaN loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.item() * v.size(0)
            train_score += batch_score

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        print("loss='%.3f'" % total_loss, "train_score='%.2f%%'" % train_score)

        logger.write('epoch %d, time %.2f' %(epoch + 1, time.time() - t))
        logger.write('train loss %.2f, train score %.2f' % (total_loss, train_score))

    model_path = os.path.join(output, '%s_%s_question_only.pth' % (dataset, preset))
    torch.save(model.state_dict(), model_path)

