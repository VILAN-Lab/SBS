import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
import numpy as np


class BaseModel_pretrain(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier):
        super(BaseModel_pretrain, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        # self.v_att = v_att
        self.q_net = q_net
        # self.v_net = v_net
        self.classifier1 = classifier
        # self.debias_loss_fn = None
        # self.bias_scale = torch.nn.Parameter(torch.from_numpy(np.ones((1, ), dtype=np.float32)*1.2))
        # self.bias_lin = torch.nn.Linear(1024, 1)

    def forward(self, v, _, q, labels, bias, return_weights=False):
        batch_size = q.size(0)
        # v = torch.ones(batch_size, 36, 2048).cuda()

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]
        q_repr = self.q_net(q_emb)

        # att = self.v_att(v, q_emb)
        # v_emb = (att * v).sum(1)
        # v_repr = self.v_net(v_emb)

        joint_repr = q_repr
        logits = self.classifier1(joint_repr)

        return logits


def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier1 = SimpleClassifier(
        num_hid, 2 * num_hid, 2274, 0.5)
    return BaseModel_repair(w_emb, q_emb, v_att, q_net, v_net, classifier1)


def build_baseline0_repair(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, 2274, 0.5)
    return BaseModel_pretrain(w_emb, q_emb, v_att, q_net, v_net, classifier)
