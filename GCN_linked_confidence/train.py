###################################################################
# File Name: train.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Thu 06 Sep 2018 10:08:49 PM CST
###################################################################
import argparse
import os
import os.path as osp
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score
from torch.backends import cudnn
from torch.utils.data import DataLoader

from GCN_linked_confidence import model
from GCN_linked_confidence.feeder.feeder import Feeder
from GCN_linked_confidence.test import euclidean_distance
from GCN_linked_confidence.utils import to_numpy
from GCN_linked_confidence.utils.logging import Logger
from GCN_linked_confidence.utils.meters import AverageMeter
from GCN_linked_confidence.utils.serialization import save_checkpoint
from GCN_linked_confidence.cluster import DensityPeakCluster
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier


# def euclidean_distance(vec1, vec2):
#     '''
#     Return the Euclidean Distance of two vectors
#     '''
#     return np.linalg.norm(vec1 - vec2)
# from sklearn.metrics.pairwise import cosine_similarity
# cosine_similarity(a)

# def build_distance_file(vectors, filename):
#     '''
#     Calculate distance, save the result for cluster
#     '''
#     with open(filename, 'w', encoding='utf-8') as outfile:
#         for i in range(len(vectors) - 1):
#             for j in range(i, len(vectors)):
#                 distance = euclidean_distance(vectors[i], vectors[j])
#                 outfile.write('%d\t%d\t%f\n' % (i + 1, j + 1, distance))
#         outfile.close()


def cllustertrain(feat_path, knn_graph_path, label_path, k_at_hop, active_connection):
    print("Training cluster...")
    logs_dir = osp.join('logs', 'logs')
    seed = 1
    workers = 16
    print_freq = 200

    # Optimization args
    lr = 1e-2
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 4

    # Training args
    batch_size = 2

    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = True
    sys.stdout = Logger(osp.join(logs_dir, 'log.txt')) # 创建文件

    trainset = Feeder(feat_path,
                      knn_graph_path,
                      label_path,
                      seed,
                      k_at_hop,
                      active_connection)
    # confidence
    gcn_num_scale = 0.3

    vectors = np.load(feat_path)
    distance_file_name = knn_graph_path + 'distance'

    if not os.path.isfile(distance_file_name):
        with open(distance_file_name, 'w', encoding='utf-8') as outfile:
            for i in range(len(vectors) - 1):
                for j in range(i, len(vectors)):
                    distance = euclidean_distance(vectors[i], vectors[j])
                    outfile.write('%d\t%d\t%f\n' % (i + 1, j + 1, distance))
            outfile.close()

    dpcluster = DensityPeakCluster()
    rho, delta = dpcluster.density_and_distance(r'' + distance_file_name)
    print(rho, len(rho), delta, len(delta))

    rho, delta, nearest_neighbor = dpcluster.cluster(20, 0.1)
    print(rho, len(rho), delta, len(delta), nearest_neighbor, len(nearest_neighbor))

    local_centrality = {}

    for i, item in enumerate(rho):  # rho from 1
        if i==0:
            continue
        local_centrality[i - 1] = item
    print(local_centrality)
    low_local_centrality_item_list = sorted(local_centrality.items(), key=lambda k: k[1],reverse=True )[
                                     0:int(gcn_num_scale * len(list(local_centrality)))]  # from low to high
    print(low_local_centrality_item_list)
    low_local_centrality_key_list = [i[0] for i in low_local_centrality_item_list]

    print('len,low_local_centrality_key_list', gcn_num_scale, len(low_local_centrality_key_list))
    trainsetSub = torch.utils.data.Subset(trainset, low_local_centrality_key_list)
    # get sub set end

    trainloader = DataLoader(
        trainsetSub, batch_size=batch_size,
        num_workers=workers, shuffle=True, pin_memory=True)

    # net = model.gcn().cuda()
    net = model.gcn()

    opt = torch.optim.SGD(net.parameters(), lr,
                          momentum=momentum,
                          weight_decay=weight_decay)

    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = nn.CrossEntropyLoss()

    save_checkpoint({
        'state_dict': net.state_dict(),
        'epoch': 0, }, False,
        fpath=osp.join(logs_dir, 'epoch_{}.ckpt'.format(0)))
    for epoch in range(epochs):
        adjust_lr(opt, epoch, lr)

        train(trainloader, net, criterion, opt, epoch, print_freq)
        save_checkpoint({
            'state_dict': net.state_dict(),
            'epoch': epoch + 1, }, False,
            fpath=osp.join(logs_dir, 'epoch_{}.ckpt'.format(epoch + 1)))


def train(loader, net, crit, opt, epoch, print_freq):
    # 初始化参数，用averageMeter
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()

    net.train()
    end = time.time()
    #  (feat, A_, center_idx, one_hop_idcs), edge_labels
    for i, ((feat, adj, cid, h1id), gtmat) in enumerate(loader):
        data_time.update(time.time() - end)
        # feat, adj, cid, h1id, gtmat = map(lambda x: x.cuda(),
        #                                   (feat, adj, cid, h1id, gtmat))
        pred = net(feat, adj, h1id)
        labels = make_labels(gtmat).long()
        loss = crit(pred, labels)
        p, r, acc = accuracy(pred, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.update(loss.item(), feat.size(0))
        accs.update(acc.item(), feat.size(0))
        precisions.update(p, feat.size(0))
        recalls.update(r, feat.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if i % print_freq == 0:
            print('Epoch:[{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                  'Accuracy {accs.val:.3f} ({accs.avg:.3f})\t'
                  'Precison {precisions.val:.3f} ({precisions.avg:.3f})\t'
                  'Recall {recalls.val:.3f} ({recalls.avg:.3f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, losses=losses, accs=accs,
                precisions=precisions, recalls=recalls))


def make_labels(gtmat):
    return gtmat.view(-1)


def adjust_lr(opt, epoch, lr):
    scale = 0.1
    print('Current lr {}'.format(lr))
    if epoch in [1, 2, 3, 4]:
        lr *= 0.1
        print('Change lr to {}'.format(lr))
        for param_group in opt.param_groups:
            param_group['lr'] = param_group['lr'] * scale


def accuracy(pred, label):
    # 返回的是索引
    pred = torch.argmax(pred, dim=1).long()
    acc = torch.mean((pred == label).float())
    pred = to_numpy(pred)
    label = to_numpy(label)
    p = precision_score(label, pred)
    r = recall_score(label, pred)
    return p, r, acc
