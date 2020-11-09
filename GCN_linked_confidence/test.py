###################################################################
# File Name: train.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Thu 06 Sep 2018 10:08:49 PM CST
###################################################################

import argparse
import os
import os.path as osp
import time

import numpy as np
import torch
import torchvision
from networkx.drawing.tests.test_pylab import plt
from sklearn.metrics import normalized_mutual_info_score, precision_score, recall_score
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms, utils
from GCN_linked_confidence import model
from GCN_linked_confidence.cluster import DensityPeakCluster
from GCN_linked_confidence.feeder.feeder import Feeder
# from GCN_linked_confidence.train import build_distance_file
from GCN_linked_confidence.utils import to_numpy
from GCN_linked_confidence.utils.graph import graph_propagation, graph_components
from GCN_linked_confidence.utils.meters import AverageMeter
from GCN_linked_confidence.utils.serialization import load_checkpoint
from GCN_linked_confidence.utils.utils import bcubed, pairwise
from torch.utils.tensorboard import SummaryWriter


# from torch_scatter import gather_csr, scatter, segment_csr

def euclidean_distance(vec1, vec2):
    '''
    Return the Euclidean Distance of two vectors
    '''
    return np.linalg.norm(vec1 - vec2)


def single_remove(Y, pred):
    single_idcs = np.zeros_like(pred)
    pred_unique = np.unique(pred)
    for u in pred_unique:
        idcs = pred == u
        if np.sum(idcs) == 1:  # if the idcs(labels) found in list are single
            single_idcs[np.where(idcs)[0][0]] = 1
    remain_idcs = [i for i in range(len(pred)) if not single_idcs[i]]
    remain_idcs = np.asarray(remain_idcs, dtype=np.int16)
    single_idc = [i for i in range(len(pred)) if single_idcs[i]]
    single_idc = np.asarray(single_idc, dtype=np.int16)
    return Y[remain_idcs], pred[remain_idcs], pred[single_idc]


def testcluster(feat_path, label_path, knn_graph_path, checkpoint, k_at_hop, active_connection, max_sz, step):
    print("Testing cluster...")
    seed = 1
    print_freq = 200

    workers = 1  # NOTE: parallel processing
    batch_size = 32  # NOTE: Divides up the dataloader total instances. Higher bs, more memory need
    #  each iteration does 2 iteration, forward and backward pass

    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = True

    # NOTE: [STEP 1 Generate IPS] Generate a sub-graph from the feature graph centered at some node, and now the
    #  sub-graph has a fixed depth, i.e. 2. Returns vertex feature, adjacent matrix, indices of the center node,
    #  unique_nodes_list, its 1-hop nodes
    valset = Feeder(feat_path,
                    knn_graph_path,
                    label_path,
                    seed,
                    k_at_hop,
                    active_connection,
                    train=False)
    # confidence
    gcn_num_scale = 0.3

    # knn_graph = np.load(knn_graph_path)
    # distances = np.load(knn_graph_path + 'distance.npy')
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

    ##ADD EDGE NEAR local_centrality CENTER
    low_local_centrality_item_list = sorted(local_centrality.items(), key=lambda k: k[1], reverse=True)[
                                     0:int(gcn_num_scale * len(list(local_centrality)))]  # from low to high

    low_local_centrality_key_list = [i[0] for i in low_local_centrality_item_list]

    print('len,low_local_centrality_key_list', gcn_num_scale, len(low_local_centrality_key_list))
    # get sub set end
    valsetSub = torch.utils.data.Subset(valset, low_local_centrality_key_list)

    # add edge near local_centrality CENTER
    high_local_centrality_edges = []
    high_local_centrality_score = []
    high_local_centrality_tree_root = 0
    for centerID, nearest_neighbor in enumerate(nearest_neighbor):  # nearest_neighbor from 1
        if centerID == 0:
            continue
        if centerID - 1 not in low_local_centrality_key_list:
            high_local_centrality_edges.append([centerID - 1, nearest_neighbor - 1])
            high_local_centrality_score.append(1)
    print('high_local_centrality_tree_root', high_local_centrality_tree_root)

    # Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    valloader = DataLoader(
        valsetSub, batch_size=batch_size,
        num_workers=workers, shuffle=False, pin_memory=True)

    ckpt = load_checkpoint(checkpoint, )
    # net = model.gcn().cuda()
    net = model.gcn()

    net.load_state_dict(ckpt['state_dict'])

    # knn_graph = valset.knn_graph
    # knn_graph_dict = list()
    # for neighbors in knn_graph:
    #     knn_graph_dict.append(dict())
    #     for n in neighbors[1:]:
    #         knn_graph_dict[-1][n] = []

    # criterion = nn.CrossEntropyLoss().cuda()
     criterion = nn.CrossEntropyLoss()

    # NOTE: [STEP 2: GCN Net]function to predict
    # edges, scores, pred_dict = validate(valloader, net, criterion, print_freq, k_at_hop)
    #
    # np.save('edges', edges)
    # np.save('scores', scores)
    # edges = np.load('edges_train.npy')
    # scores = np.load('scores_train.npy')

    new_edges = []
    new_scores = []
    new_edges.extend(high_local_centrality_edges)
    new_scores.extend(high_local_centrality_score)

    tree_root_count = 0
    net.eval()
    # get pred again
    # node_list包含一阶二阶邻居节点
    for i, ((feat, adj, cid, h1id, node_list), gtmat) in enumerate(valloader):
        # feat, adj, cid, h1id, gtmat = map(lambda x: x.cuda(),
        #                                   (feat, adj, cid, h1id, gtmat))
        pred = net(feat, adj, h1id)
        pred = F.softmax(pred, dim=1)
        # pred = pred_dict[i]
        node_list = node_list.long().squeeze().numpy()
        bs = feat.size(0)

        for b in range(bs):
            cidb = cid[b].int().item()
            nl = node_list[b]
            if len(nl.shape) == 0:
                print(cidb, nl, 'len(nl.shape) == 0')
                continue

            temp_max_score = 0  #
            temp_max_id = -1

            for j, n in enumerate(h1id[b]):
                n = n.item()

                if nl[cidb] in local_centrality and nl[n] in local_centrality:
                    if local_centrality[nl[cidb]] < local_centrality[nl[n]]:
                        if pred[b * k_at_hop[0] + j, 1].item() > temp_max_score:
                            temp_max_score = pred[b * k_at_hop[0] + j, 1]
                            temp_max_id = nl[n]
                else:
                    print('local_centrality INDEX', nl[cidb], nl[n], cidb, n, nl.shape, nl)

            if temp_max_id != -1:
                if nl[cidb] in low_local_centrality_key_list:
                    new_edges.append([nl[cidb], temp_max_id])
                    # we need change this prediction score （pred[b * args.k_at_hop[0] + j, 1].item()）
                    # to feature cosine between two instance
                    new_scores.append(0)
                    # print(cosine_similarity(feat)[[nl[cidb], temp_max_id]])
                else:
                    new_edges.append([nl[cidb], temp_max_id])
                    # we need change this prediction score （pred[b * args.k_at_hop[0] + j, 1].item()）
                    # to feature cosine between two instance
                    new_scores.append(temp_max_score)
                    # print(cosine_similarity(feat)[[nl[cidb], temp_max_id]])

            else:
                new_edges.append([nl[cidb], nl[cidb]])
                new_scores.append(0)
                tree_root_count += 1

    print('tree_root_count', tree_root_count)

    new_edges = np.asarray(new_edges)
    new_scores = np.asarray(new_scores)

    temp = []
    for item in new_edges:
        temp.append(item[0])
        temp.append(item[0])

    print('set(new_edges[:0]+new_edges[:0] total unique', len(set(temp)))

    clusters = graph_components(new_edges, new_scores, th=0.00)
    print('clusters', len(clusters))

    final_pred = clusters2labels(clusters, len(valset))
    print('final_pred', len(final_pred))
    labels = valset.labels

    print('------------------------------------')
    print('Number of nodes: ', len(labels), 'class num', len(set(final_pred)))
    print('bcubed_Precision   Recall   F-Sore   NMI PairWise_Precision   Recall   F-Sore')

    p, r, f = bcubed(labels, final_pred)
    nmi = normalized_mutual_info_score(final_pred, labels)

    pw_p, pw_r, pw_f = pairwise(final_pred, labels)
    print(('{:.4f}    ' * 7).format(p, r, f, nmi, pw_p, pw_r, pw_f))

    labels, final_pred, singleton_pred = single_remove(labels, final_pred)

    print('------------------------------------')
    print('After removing singleton culsters, number of nodes: ', len(labels), 'class num', len(set(final_pred)))
    print('bcubed_Precision   Recall   F-Sore   NMI PairWise_Precision   Recall   F-Sore')
    p, r, f = bcubed(labels, final_pred)
    nmi = normalized_mutual_info_score(final_pred, labels)
    pw_p, pw_r, pw_f = pairwise(final_pred, labels)
    print(('{:.4f}    ' * 7).format(p, r, f, nmi, pw_p, pw_r, pw_f))

    ######new add end here


def clusters2labels(clusters, n_nodes):
    labels = (-1) * np.ones((n_nodes,))
    for ci, c in enumerate(clusters):
        for xid in c:
            labels[xid.name] = ci
    assert np.sum(labels < 0) < 1
    return labels


def make_labels(gtmat):
    return gtmat.view(-1)


def validate(loader, net, crit, print_freq, k_at_hop):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()

    net.eval()
    end = time.time()
    edges = list()
    scores = list()
    pred_dict = {}
    for i, ((feat, adj, cid, h1id, node_list), gtmat) in enumerate(
            loader):  # iterate over each instances and its properties

        data_time.update(time.time() - end)
        # feat, adj, cid, h1id, gtmat = map(lambda x: x.cuda(),
        #                                   (feat, adj, cid, h1id, gtmat))  # Map arrays to GPU/cuda memory

        pred = net(feat, adj, h1id)

        labels = make_labels(gtmat).long()
        loss = crit(pred, labels)
        pred = F.softmax(pred, dim=1)

        # pred_dict[i] = pred

        p, r, acc = accuracy(pred, labels)

        losses.update(loss.item(), feat.size(0))
        accs.update(acc.item(), feat.size(0))
        precisions.update(p, feat.size(0))
        recalls.update(r, feat.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if i % print_freq == 0:
            print('[{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {losses.val:.3f} ({losses.avg:.3f})\n'
                  'Accuracy {accs.val:.3f} ({accs.avg:.3f})\t'
                  'Precison {precisions.val:.3f} ({precisions.avg:.3f})\t'
                  'Recall {recalls.val:.3f} ({recalls.avg:.3f})'.format(
                i, len(loader), batch_time=batch_time,
                data_time=data_time, losses=losses, accs=accs,
                precisions=precisions, recalls=recalls))

        node_list = node_list.long().squeeze().numpy()

        bs = feat.size(0)
        for b in range(bs):
            cidb = cid[b].int().item()
            nl = node_list[b]

            for j, n in enumerate(h1id[b]):
                n = n.int().item()
                if len(nl.shape) > 0:
                    edges.append([nl[cidb], nl[n]])
                    scores.append(pred[b * k_at_hop[0] + j, 1].item())
                else:
                    print(i, 'cidb, nl, [nl[cidb], nl[n]], n', cidb, nl, type(cidb), type(n), n, type(nl), nl.shape)
                    print('nlnl.shape=0')
    edges = np.asarray(edges)
    scores = np.asarray(scores)
    return edges, scores, pred_dict


def accuracy(pred, label):
    pred = torch.argmax(pred, dim=1).long()
    acc = torch.mean((pred == label).float())
    pred = to_numpy(pred)
    label = to_numpy(label)
    p = precision_score(label, pred)
    r = recall_score(label, pred)
    return p, r, acc
