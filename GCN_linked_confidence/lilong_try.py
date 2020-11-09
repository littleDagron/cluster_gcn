import os
# path = "D:/tmp/home"
# os.makedirs(path)
# print("success")
# a = {11:'a', 5:'b', 4:'c',}
# un_map = [i for i in a]
# print(un_map)
import numpy as np
dirpath = "D:/gcn/DPC+linkageGCN/GCN_linked_confidence/data/knns/deepfashion_train/faiss_k_5.npz"
data_train = np.load(dirpath)
data_t = data_train['data']
print(data_t.shape)
# def npfromfile2load(npbinpath, labelmetapath, dim, datapath, labelpath):
#     label_train = [int(x) for x in open(labelmetapath, 'r').readlines()]  # save meta as np
#     totins = len(label_train)
#     print(totins)
#     dimxins = totins * dim
#     feature_train = np.fromfile(npbinpath, dtype=np.float32, count=dimxins).reshape(totins, dim)
#     np.save(datapath, feature_train)  # save bin as np
#
#     np.savetxt(labelpath, label_train)
#
#
# dim = 256
# npbinpath = "D:/gcn/DPC+linkageGCN/GCN_linked_confidence/data/features/deepfashion_train.bin"
# labelmetapath = "D:/gcn/DPC+linkageGCN/GCN_linked_confidence/data/labels/deepfashion_train.meta"
# datapath = "D:/gcn/DPC+linkageGCN/GCN_linked_confidence/data/features/datadeepfashion_train.npy"
# labelpath = "D:/gcn/DPC+linkageGCN/GCN_linked_confidence/data/labels/deepfashion_train.npy"
#
# npfromfile2load(npbinpath, labelmetapath, dim, datapath, labelpath)