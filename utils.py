from __future__ import print_function

import errno
import os
import numpy as np
# from PIL import Image
import torch
import torch.nn as nn
from collections import defaultdict, Counter

EPS = 1e-7


def calculate_bias(train_dset, answer_voc_size, unbias_sub_dset, bias_sub_dset):
    question_type_to_probs = defaultdict(Counter)
    question_type_to_count = Counter()
    for ex in train_dset.entries:
        ans = ex["answer"]
        q_type = ans["question_type"]
        question_type_to_count[q_type] += 1
        if ans["labels"] is not None:
            for label, score in zip(ans["labels"], ans["scores"]):
                question_type_to_probs[q_type][label] += score

    question_type_to_prob_array = {}
    for q_type, count in question_type_to_count.items():
        prob_array = np.zeros(answer_voc_size, np.float32)
        for label, total_score in question_type_to_probs[q_type].items():
            prob_array[label] += total_score
        prob_array /= count
        # which:          [ 0.1, 0.2, 0.2, 0.1, ..., 0.1, 0.1]
        # what color is   [ 0.2, 0.1, 0.3, 0.1, ..., 0.2, 0.1]
        question_type_to_prob_array[q_type] = prob_array

    for ex in unbias_sub_dset.entries:
        q_type = ex["answer"]["question_type"]
        ex["bias"] = question_type_to_prob_array[q_type]

    for ex in bias_sub_dset.entries:
        q_type = ex["answer"]["question_type"]
        ex["bias"] = question_type_to_prob_array[q_type]


def get_keep_idx(w, cls_idx, select_ratio, mode='threshold'):

    # strategy 1: top k% examples
    if mode == 'rank':
        keep_examples = int(round(select_ratio * len(w)))
        keep_idx = w.sort(descending=True)[1][:keep_examples].cuda()
        abandon_idx = w.sort(descending=True)[1][keep_examples:].cuda()

    # strategy 2: top k% examples each class
    elif mode == 'cls_rank':
        keep_idx_list = []
        abandon_idx_list= []
        for c in range(2274):
            c_idx = cls_idx[c].nonzero()
            keep_examples = round(select_ratio * len(c_idx))
            c_idx = c_idx.squeeze()
            sort_idx = w[c_idx].sort(descending=True)[1]
            keep_idx_list.append(c_idx[sort_idx][:keep_examples])
            abandon_idx_list.append(c_idx[sort_idx][keep_examples:])
        keep_idx = torch.cat(keep_idx_list).cuda()
        abandon_idx = torch.cat(abandon_idx_list).cuda()

    # strategy 3: random uniform sampling
    elif mode == 'uniform':
        keep_examples = round(select_ratio * len(w))
        keep_idx = torch.randperm(len(w))[:keep_examples]

    return keep_idx, abandon_idx


def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)


def assert_array_eq(real, expected):
    assert (np.abs(real-expected) < EPS).all(), \
        '%s (true) vs %s (expected)' % (real, expected)


def load_folder(folder, suffix):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(suffix):
            imgs.append(os.path.join(folder, f))
    return imgs


# def load_imageid(folder):
#     images = load_folder(folder, 'jpg')
#     img_ids = set()
#     for img in images:
#         img_id = int(img.split('/')[-1].split('.')[0].split('_')[-1])
#         img_ids.add(img_id)
#     return img_ids


# def pil_loader(path):
#     with open(path, 'rb') as f:
#         with Image.open(f) as img:
#             return img.convert('RGB')


def weights_init(m):
    """custom weights initialization."""
    cname = m.__class__
    if cname == nn.Linear or cname == nn.Conv2d or cname == nn.ConvTranspose2d:
        m.weight.data.normal_(0.0, 0.02)
    elif cname == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        print('%s is not initialized.' % cname)


def init_net(net, net_file):
    if net_file:
        net.load_state_dict(torch.load(net_file))
    else:
        net.apply(weights_init)


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


class Logger(object):
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        print(msg)
