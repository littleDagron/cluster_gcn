import os

from GCN_linked_confidence.test import testcluster
from GCN_linked_confidence.train import cllustertrain
from GCN_linked_confidence.utils.utils import getknn_brute
# from misc import npfromfile2load
# from misc import npfromfile2load
import numpy as np


def npfromfile2load(npbinpath, labelmetapath, dim, datapath, labelpath):
    label_train = [int(x) for x in open(labelmetapath, 'r').readlines()]  # save meta as np
    totins = len(label_train)
    print(totins)
    dimxins = totins * dim
    feature_train = np.fromfile(npbinpath, dtype=np.float32, count=dimxins).reshape(totins, dim)
    np.save(datapath, feature_train)  # save bin as np

    np.savetxt(labelpath, label_train)


trainbinpath = "../data/features/wos_linkedbased_feature_trainset.bin"
trainmetapath = "../data/labels/wos_linkedbased_label_trainset.meta"

train_feat = "../data/features/wos_linkedbased_feature_trainset.npy"
train_label = "../data/labels/wos_linkedbased_label_trainset.npy"
train_knn = "../data/knns/wos_linkedbased_knn_trainset.npy"

testbinpath = "../data/features/wos_linkedbased_feature_testset.bin"
testmetapath = "../data/labels/wos_linkedbased_label_testset.meta"

test_feat = "../data/features/wos_linkedbased_feature_testset.npy"
test_label = "../data/labels/wos_linkedbased_label_testset.npy"
test_knn = "../data/knns/wos_linkedbased_knn_testset.npy"

dim = 100
n_neighbors = 41
k_at_hop = [40, 10]
active_connection = 10
max_sz = 900  # maximal size of a cluster(knns threshold for each node), more max = less pre, more re
step = 0.6  # the step to adjust threshold, default: 0.6, # recommended range: [0.3, 0.8], more th --> low re and more pre

# # trainbinpath = "../data/features/deepfashion_train.bin"
# # trainmetapath = "../data/labels/deepfashion_train.meta"
# #
# train_feat = "../data/features/deepfashion_train.npy"
# train_label = "../data/labels/deepfashion_train.npy"
# train_knn = "../data/knns/deepfashion_train.npy"
# #
# # testbinpath = "../data/features/deepfashion_test.bin"
# # testmetapath = "../data/labels/deepfashion_test.meta"
# #
# test_feat = "../data/features/deepfashion_test.npy"
# test_label = "../data/labels/deepfashion_test.npy"
# test_knn = "../data/knns/deepfashion_test.npy"
# dim = 256
# n_neighbors = 6
# k_at_hop = [5, 5]
# active_connection = 5
# max_sz = 40 #maximal size of a cluster(knns threshold for each node), more max = less pre, more re
# step = 0.5  # the step to adjust threshold, default: 0.6, # recommended range: [0.3, 0.8], more th --> low re and more pre
#
#
# test_feat = "../data/features/512.fea.npy"
# test_label = "../data/labels/512.labels.npy"
# test_knn = "../data/knns/knn.graph.512.bf.npy"
#
#
# # test_feat = "../data/features/1024.fea.npy"
# # test_label = "../data/labels/1024.labels.npy"
# # test_knn = "../data/knns/knn.graph.1024.bf.npy"
# #
# # test_feat = "../data/features/1845.fea.npy"
# # test_label = "../data/labels/1845.labels.npy"
# # test_knn = "../data/knns/knn.graph.1845.bf.npy"
#
# # train_feat = "../data/features/1845.fea.npy"
# # train_label = "../data/labels/1845.labels.npy"
# # train_knn = "../data/knns/knn.graph.1845.bf.npy"
# #
# train_feat = "../data/features/512.fea.npy"
# train_label = "../data/labels/512.labels.npy"
# train_knn = "../data/knns/knn.graph.512.bf.npy"
#
# dim = 512
# n_neighbors = 21
# k_at_hop = [20, 5]
# active_connection = 5
# max_sz = 900 #maximal size of a cluster(knns threshold for each node), more max = less pre, more re
# step = 0.6  # the step to adjust threshold, default: 0.6, # recommended range: [0.3, 0.8], more th --> low re and more pre

checkpoint = './logs/logs/epoch_4.ckpt'
# checkpoint = './logs/logs/best.ckpt'

# dim = 256
# n_neighbors = 141
# k_at_hop = [140, 10]
# active_connection = 10
#
# npfromfile2load(trainbinpath, trainmetapath, dim, train_feat, train_label)
# getknn_brute(train_feat, train_label, train_knn, n_neighbors)
# cllustertrain(train_feat, train_knn, train_label, k_at_hop, active_connection)

# n_neighbors = 101
# k_at_hop = [100, 10]
# active_connection = 10
# max_sz = 500  # maximal size of a cluster(knns threshold for each node), more max = less pre, more re
# step = 0.1  # the step to adjust threshold, default: 0.6, # recommended range: [0.3, 0.8], more th --> low re and more pre

# npfromfile2load(trainbinpath, trainmetapath, dim, train_feat, train_label)
# getknn_brute(train_feat, train_label, train_knn, n_neighbors)
# cllustertrain(train_feat, train_knn, train_label, k_at_hop, active_connection)

# NOTE: deepfashion total:
#  train = 25752, total labels = 3990
#  test = 26960, total labels = 3990

# npfromfile2load(testbinpath, testmetapath, dim, test_feat, test_label)
# getknn_brute(test_feat, test_label, test_knn, n_neighbors)
# testcluster(test_feat, test_label, test_knn, checkpoint, k_at_hop, active_connection, max_sz, step)

if __name__ == "__main__":
    cllustertrain(train_feat, train_knn, train_label, k_at_hop, active_connection)
    testcluster(test_feat, test_label, test_knn, checkpoint, k_at_hop, active_connection, max_sz, step)
