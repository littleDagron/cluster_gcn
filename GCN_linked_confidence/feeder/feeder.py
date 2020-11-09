 ###################################################################
# File Name: feeder.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Thu 06 Sep 2018 01:06:16 PM CST
###################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import torch
import torch.utils.data as data


class Feeder(data.Dataset):
    '''
    Generate a sub-graph from the feature graph centered at some node, 
    and now the sub-graph has a fixed depth, i.e. 2
    '''

    def __init__(self, feat_path, knn_graph_path, label_path, seed=1,
                 k_at_hop=[200, 5], active_connection=5, train=True):
        np.random.seed(seed)
        random.seed(seed)
        self.features = np.load(feat_path)  # (36092,512) 1024.fea.npy
        print(feat_path)
        self.knn_graph = np.load(knn_graph_path)[:, :k_at_hop[0] + 1]
        # self.labels = np.loadtxt(label_path)
        self.labels = np.load(label_path)

        self.num_samples = len(self.features)  # NOTE: number of instances
        self.depth = len(k_at_hop)  # NOTE:depth tree of 2
        self.k_at_hop = k_at_hop  # NOTE: no. of hops
        self.active_connection = active_connection  # NOTE: to find out

        self.train = train
        assert np.mean(k_at_hop) >= active_connection

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # NOTE: index passed in the iteration, depends on worker, batch size. In ascending order
        '''
        return the vertex feature and the adjacent matrix A, together 
        with the indices of the center node and its 1-hop nodes
        '''
        # hops[0] for 1-hop neighbors, hops[1] for 2-hop neighbors
        hops = list()  # it is just the knn of each instances
        center_node = index  # making each instances as the center node
        hops.append(set(self.knn_graph[center_node][
                        1:]))  # NOTE: get the knn of the instances, [1:0] is to get the knn other than its owm
        # 一阶邻居
        # Actually we dont need the loop since the depth is fixed here,
        # But we still remain the code for further revision

        for d in range(1, self.depth):  # range of depth =2
            hops.append(set())  # initiate 2 set tuples of each depth of hops
            for h in hops[0]:  # for each neighbours in first hop list
                # 添加二阶邻居
                hops[1].update(set(self.knn_graph[h][1:self.k_at_hop[
                                                           d] + 1]))  # NOTE: append list of 2nd hops neighbours for each first hop neaigbours
        # 对于每个节点，创建一阶二阶邻居节点集合
        hops_set = set([h for hop in hops for h in hop])  # for each hops, get unique list of nightbours

        hops_set.update([center_node, ])

        unique_nodes_list = list(hops_set)
        # i是索引index
        unique_nodes_map = {j: i for i, j in enumerate(unique_nodes_list)} # dict of index of hop list and index,accending order
        # center_idx确定中心点所在的位置（索引）
        center_idx = torch.Tensor([unique_nodes_map[center_node], ]).type(torch.long)
        # 一节邻居的索引
        one_hop_idcs = torch.Tensor([unique_nodes_map[i] for i in hops[0]]).type(torch.long)
        # get the features of the nodes of currecnt instance/pivot
        # 中心节点的特征
        center_feat = torch.Tensor(self.features[center_node]).type(torch.float)
        # get the features of the nodes in hops_neighbours of current instance/pivot
        # 当前中心节点所有邻居的特征，包括中心点
        feat = torch.Tensor(self.features[unique_nodes_list]).type(torch.float)
        # node feature normalised by substracting the features on hops neighbouts and instance/pivot feature
        # feat特征正则化，减去中心节点的特征
        feat = feat - center_feat

        max_num_nodes = self.k_at_hop[0] * (self.k_at_hop[1] + 1) + 1  # max nodes possible using hops val up to depth
        # 当前中心节点的所有邻居节点的数量，包括一阶二阶
        num_nodes = len(unique_nodes_list)  # length of hops neighbours, total edges
        # 邻接矩阵
        A = torch.zeros(num_nodes, num_nodes)  # initialise adj matrix
        # fdim是特征维度512
        _, fdim = feat.shape
        # features of remaining nodes not in hops neighbour
        feat = torch.cat([feat, torch.zeros(max_num_nodes - num_nodes, fdim)], dim=0)

        for node in unique_nodes_list: # creats adjecent matrix based on active node connection with hops neighbour
            # get the first n-neighbours up till active_connection val
            neighbors = self.knn_graph[node, 1:self.active_connection + 1]
            for n in neighbors:
                if n in unique_nodes_list: # if the neigbours picked from a_c has neighbour in hops neig
                    A[unique_nodes_map[node], unique_nodes_map[n]] = 1
                    A[unique_nodes_map[n], unique_nodes_map[node]] = 1

        # get average weight from adj matrix and store in each adj grid
        D = A.sum(1, keepdim=True)
        # A除以D，逐个元素相除
        A = A.div(D)
        A_ = torch.zeros(max_num_nodes, max_num_nodes)
        A_[:num_nodes, :num_nodes] = A

        labels = self.labels[np.asarray(unique_nodes_list)] # get the labels of the hops neighbours
        labels = torch.from_numpy(labels).type(torch.long)
        # edge_labels = labels.expand(num_nodes,num_nodes).eq(
        #        labels.expand(num_nodes,num_nodes).t())
        # 一节邻居的label
        one_hop_labels = labels[one_hop_idcs] # get hop labels
        # 中心节点（中枢pivot）的label
        center_label = labels[center_idx] # get pivot/instance label
        # center_label和one_hop_labels中的元素逐个比较，相同返回1
        edge_labels = (center_label == one_hop_labels).long() # get linked edge onehotencode to pivot/instnaces
        # 返回feat特征， A_邻接矩阵， 中心点索引， 一节邻居索引， 边是否存在标记
        if self.train:
            return (feat, A_, center_idx, one_hop_idcs), edge_labels # normalised feature, adj matrix, pivot index, hops

        # Testing
        # unique_nodes_list表示一阶二阶邻居节点
        unique_nodes_list = torch.Tensor(unique_nodes_list).long()
        unique_nodes_list = torch.cat(
            [unique_nodes_list, torch.zeros(max_num_nodes - num_nodes)], dim=0)
        return (feat, A_, center_idx, one_hop_idcs, unique_nodes_list), edge_labels
