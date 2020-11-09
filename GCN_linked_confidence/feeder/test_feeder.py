###################################################################
# File Name: test_feeder.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Thu 06 Sep 2018 04:09:46 PM CST
###################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from GCN_linked_confidence.feeder.feeder import Feeder

torch.set_printoptions(threshold=10000000, linewidth=500)
if __name__ == '__main__':
    feeder = Feeder('../../facedata/1024.fea.npy',
                    '../../facedata/knns.graph.1024.kdtree.npy',
                    '../../facedata/1024.labels.npy',
                    seed=2111112,
                    k_at_hop=[5, 5],
                    active_connection=3)
    (feat, A, cn, oh), edge_labels = feeder[0]

    print(oh)
    print(edge_labels)
    length = feat.norm(2, dim=1)
    print(length)
    # print(torch.sum(A,dim=1))
    # print(torch.sum(edge_labels,dim=0))
    # print(A)
    # print(edge_labels)
    # print(feat.shape)
    # feat =feat.div(feat.norm(2,dim=1,keepdim=True).expand_as(feat))
    # print(torch.mm(feat,feat.t()))
