import argparse
import json
import pickle
from collections import defaultdict, Counter
from os.path import dirname, join
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from select_data import select

from dataset import Dictionary, VQAFeatureDataset, IndexedDataset
from dataset_select import VQAFeatureDataset_Select
from dataset_conca import VQAFeatureDataset_Conca
import base_model
import Model_repair
import model_pretrain
from train import train
import utils
import click
from pretrain import pretrain
from train_bias import train_bias
from train_unbias import train_unbias
from repair_data import repair_data
from utils import calculate_bias
from vqa_debias_loss_functions import *


def parse_args():
    parser = argparse.ArgumentParser("Train the BottomUpTopDown model with a de-biasing method")

    # Arguments we added
    parser.add_argument(
        '--cache_features', default=False,
        help="Cache image features in RAM. Make s things much faster, "
             "especially if the filesystem is slow, but requires at least 48gb of RAM")
    parser.add_argument(
        '--dataset', default='cpv2',
        choices=["v2", "cpv2", "cpv1"],
        help="Run on VQA-2.0 instead of VQA-CP 2.0"
    )
    parser.add_argument(
        '-p', "--entropy_penalty", default=0.36, type=float,
        help="Entropy regularizer weight for the learned_mixin model")
    parser.add_argument(
        '--mode', default="updn",
        choices=["updn", "q_debias", "v_debias", "q_v_debias"],
        help="Kind of ensemble loss to use")
    parser.add_argument(
        '--debias', default="learned_mixin",
        choices=["learned_mixin", "reweight", "bias_product", "none", 'focal'],
        help="Kind of ensemble loss to use")
    parser.add_argument(
        '--topq', type=int,default=1,
        choices=[1,2,3],
        help="num of words to be masked in questio")
    parser.add_argument(
        '--keep_qtype', default=True,
        help="keep qtype or not")
    parser.add_argument(
        '--topv', type=int,default=-1,
        choices=[1,3,5,-1],
        help="num of object bbox to be masked in image")
    parser.add_argument(
        '--top_hint',type=int, default=9,
        choices=[9,18,27,36],
        help="num of hint")
    parser.add_argument(
        '--qvp', type=int,default=5,
        choices=[0,1,2,3,4,5,6,7,8,9,10],
        help="ratio of q_bias and v_bias")
    parser.add_argument(
        '--eval_each_epoch', default=True,
        help="Evaluate every epoch, instead of at the end")

    # Arguments from the original model, we leave this default, except we
    # set --epochs to 30 since the model maxes out its performance on VQA 2.0 well before then
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='logs/exp0')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--sampling', default='rank',
                        choices=['rank', 'cls_rank', 'uniform'], type=str)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--loss_weight', type=float, default=0.7)
    parser.add_argument('--preset', default='trainval',
                        choices=['train', 'val', 'trainval'])
    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    dataset = args.dataset
    args.output = os.path.join('logs', args.output)
    # if not os.path.isdir(args.output):
    #     utils.create_dir(args.output)
    # else:
    #     if click.confirm('Exp directory already exists in {}. Erase?'
    #                              .format(args.output, default=False)):
    #         os.system('rm -r ' + args.output)
    #         utils.create_dir(args.output)
    #     else:
    #         os._exit(1)

    if dataset=='cpv1':
        dictionary = Dictionary.load_from_file('data/dictionary_v1.pkl')
    if dataset=='cpv2' or dataset=='v2':
        dictionary = Dictionary.load_from_file('data/dictionary.pkl')

    with open('util/qid2type_%s.json' % args.dataset, 'r') as f:
        qid2type = json.load(f)


    # print("Building train dataset for repair...")
    # if args.preset == 'train':
    #     pretrain_dset = VQAFeatureDataset('train', dictionary, dataset=dataset,
    #                                cache_image_features=args.cache_features)
    # elif args.preset == 'val':
    #     pretrain_dset = VQAFeatureDataset('val', dictionary, dataset=dataset,
    #                                cache_image_features=args.cache_features)
    # elif args.preset == 'trainval':
    #     print("loading v2 train and test set")
    #     pretrain_dset = VQAFeatureDataset_Conca('train', dictionary, dataset="v2",
    #                                    cache_image_features=True)
    # pre_loader1 = DataLoader(pretrain_dset, args.batch_size, shuffle=True, num_workers=0)
    # pre_dset = VQAFeatureDataset('train', dictionary, dataset="cpv2",
    #                                       cache_image_features=False)
    # pre_loader2 = DataLoader(pre_dset, args.batch_size, shuffle=True, num_workers=0)

    train_dset = VQAFeatureDataset_Select('train', dictionary, dataset=dataset,
                                   cache_image_features=True)

    select_dset_idx = IndexedDataset(train_dset)
    select_loader = DataLoader(select_dset_idx, args.batch_size, shuffle=False, num_workers=0)


    # print('Starting pretrain model...')
    pre_output = 'saved_models/pretrain'
    # model_repair = getattr(Model_repair, 'build_baseline0_repair')(train_dset, args.num_hid).cuda()
    # pretrain(model_repair, pre_loader1, 20, pre_output, "v2", "trainval")
    # print('pretrain End...')


    print('Loading pretrain model...')
    # pretrain_path = os.path.join(pre_output, '%s_%s_question_only.pth' % (args.dataset, args.preset))
    #
    # pre_model1_state = torch.load(pretrain_path)
    pre_model2 = getattr(model_pretrain, 'build_baseline0_repair')(train_dset, args.num_hid).cuda()
    # pre_model2_dict = pre_model2.state_dict()
    # load_state_dict = {k:v for k,v in pre_model1_state.items() if k in pre_model2_dict.keys()}
    # pre_model2_dict.update(load_state_dict)
    # pre_model2.load_state_dict(pre_model2_dict)
    pre_model2 = pre_model2.cuda()
    #
    # pretrain(pre_model2, pre_loader2, 20, pre_output, "cpv2", "train", ispre2=True)

    final_state_dict = os.path.join(pre_output, "cpv2_train_question_only.pth")
    pre_model2.load_state_dict(torch.load(final_state_dict))
    select_ratio = select(pre_model2, select_loader, qid2type)


    with open('list_keep.json', 'r') as list_r:
        keep_idx = json.load(list_r)
    with open('list_abandon.json', 'r') as list_r:
        abandon_idx = json.load(list_r)


    print("Building test dataset...")
    eval_dset = VQAFeatureDataset('val', dictionary, dataset=dataset,
                                  cache_image_features=True)
    torch.cuda.empty_cache()
    print("Building unbias and bias dataset")
    unbias_dset = VQAFeatureDataset('train', dictionary, dataset=dataset,
                                  cache_image_features=True, keepidx=keep_idx)
    unbias_loader = DataLoader(unbias_dset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    bias_dset = VQAFeatureDataset('train', dictionary, dataset=dataset,
                                  cache_image_features=True, keepidx=abandon_idx)
    # bias_batch = int(round(args.batch_size * (1 - select_ratio) / select_ratio))
    batch_size_bias = 0.8
    bias_batch = int(round(args.batch_size * (1 - batch_size_bias) / batch_size_bias))
    bias_loader = DataLoader(bias_dset, batch_size=bias_batch, shuffle=True, num_workers=0)
    torch.cuda.empty_cache()

    print("Calculate bias")
    answer_voc_size = train_dset.num_ans_candidates
    calculate_bias(train_dset, answer_voc_size, unbias_dset, bias_dset)

    # Build the model using the original constructor
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
    if dataset=='cpv1':
        model.w_emb.init_embedding('data/glove6b_init_300d_v1.npy')
    elif dataset=='cpv2' or dataset=='v2':
        model.w_emb.init_embedding('data/glove6b_init_300d.npy')

    model.debias_loss_fn = LearnedMixin(args.entropy_penalty)
    model = model.cuda()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    eval_loader = DataLoader(eval_dset, args.batch_size, shuffle=False, num_workers=0)

    print("Starting training...")
    torch.cuda.empty_cache()
    train(model, unbias_loader, bias_loader, eval_loader, args, qid2type, select_ratio)


if __name__ == '__main__':
    main()






