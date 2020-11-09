###################################################################
# File Name: utils.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Tue 28 Aug 2018 04:57:29 PM CST
###################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pprint import pprint

import matplotlib
import numpy as np
from joblib.numpy_pickle_utils import xrange
from scipy.sparse import coo_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def norm(X):
    for ix, x in enumerate(X):
        X[ix] /= np.linalg.norm(x)
    return X


def plot_embedding(X, Y):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure(figsize=(10, 10))
    for i in xrange(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(Y[i]),
                 color=plt.cm.Set1(Y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 12})
    plt.savefig('a.jpg')


EPS = np.finfo(float).eps


def Bcubed_contingency_matrix(ref_labels, sys_labels):
    """Return contingency matrix between ``ref_labels`` and ``sys_labels``."""
    ref_classes, ref_class_inds = np.unique(ref_labels, return_inverse=True)
    sys_classes, sys_class_inds = np.unique(sys_labels, return_inverse=True)
    n_frames = ref_labels.size
    # Following works because coo_matrix sums duplicate entries. Is roughly
    # twice as fast as np.histogram2d.
    cmatrix = coo_matrix(
        (np.ones(n_frames), (ref_class_inds, sys_class_inds)),
        shape=(ref_classes.size, sys_classes.size),
        dtype=np.int)
    cmatrix = cmatrix.toarray()
    return cmatrix, ref_classes, sys_classes


from scipy import sparse as sp


def contingency_matrix(labels_true, labels_pred, eps=None, sparse=False):
    """Build a contingency matrix describing the relationship between labels.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference

    labels_pred : array, shape = [n_samples]
        Cluster labels to evaluate

    eps : None or float, optional.
        If a float, that value is added to all values in the contingency
        matrix. This helps to stop NaN propagation.
        If ``None``, nothing is adjusted.

    sparse : boolean, optional.
        If True, return a sparse CSR continency matrix. If ``eps is not None``,
        and ``sparse is True``, will throw ValueError.

        .. versionadded:: 0.18

    Returns
    -------
    contingency : {array-like, sparse}, shape=[n_classes_true, n_classes_pred]
        Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
        true class :math:`i` and in predicted class :math:`j`. If
        ``eps is None``, the dtype of this array will be integer. If ``eps`` is
        given, the dtype will be float.
        Will be a ``scipy.sparse.csr_matrix`` if ``sparse=True``.
    """

    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")

    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    contingency = sp.coo_matrix((np.ones(class_idx.shape[0]),
                                 (class_idx, cluster_idx)),
                                shape=(n_classes, n_clusters),
                                dtype=np.int)
    if sparse:
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
    else:
        contingency = contingency.toarray()
        if eps is not None:
            # don't use += as contingency is integer
            contingency = contingency + eps
    return contingency


def bcubed(ref_labels, sys_labels, cm=None):
    """Return B-cubed precision, recall, and F1.

    The B-cubed precision of an item is the proportion of items with its
    system label that share its reference label (Bagga and Baldwin, 1998).
    Similarly, the B-cubed recall of an item is the proportion of items
    with its reference label that share its system label. The overall B-cubed
    precision and recall, then, are the means of the precision and recall for
    each item.

    Parameters
    ----------
    ref_labels : ndarray, (n_frames,)
        Reference labels.

    sys_labels : ndarray, (n_frames,)
        System labels.

    cm : ndarray, (n_ref_classes, n_sys_classes)
        Contingency matrix between reference and system labelings. If None,
        will be computed automatically from ``ref_labels`` and ``sys_labels``.
        Otherwise, the given value will be used and ``ref_labels`` and
        ``sys_labels`` ignored.
        (Default: None)

    Returns
    -------
    precision : float
        B-cubed precision.

    recall : float
        B-cubed recall.

    f1 : float
        B-cubed F1.

    References
    ----------
    Bagga, A. and Baldwin, B. (1998). "Algorithms for scoring coreference
    chains." Proceedings of LREC 1998.
    """
    if cm is None:
        cm, _, _ = Bcubed_contingency_matrix(ref_labels, sys_labels)
    cm = cm.astype('float64')
    cm_norm = cm / cm.sum()
    precision = np.sum(cm_norm * (cm / cm.sum(axis=0)))
    recall = np.sum(cm_norm * (cm / np.expand_dims(cm.sum(axis=1), 1)))
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def pairwise(gt_labels, pred_labels, sparse=True):
    _check(gt_labels, pred_labels)
    return fowlkes_mallows_score(gt_labels, pred_labels, sparse)


def _compute_fscore(pre, rec):
    return 2. * pre * rec / (pre + rec)


def fowlkes_mallows_score(gt_labels, pred_labels, sparse=True):
    ''' The original function is from `sklearn.metrics.fowlkes_mallows_score`.
        We output the pairwise precision, pairwise recall and F-measure,
        instead of calculating the geometry mean of precision and recall.
    '''
    n_samples, = gt_labels.shape

    c = contingency_matrix(gt_labels, pred_labels, sparse=sparse)
    tk = np.dot(c.data, c.data) - n_samples
    pk = np.sum(np.asarray(c.sum(axis=0)).ravel() ** 2) - n_samples
    qk = np.sum(np.asarray(c.sum(axis=1)).ravel() ** 2) - n_samples

    avg_pre = tk / pk
    avg_rec = tk / qk
    fscore = _compute_fscore(avg_pre, avg_rec)

    return avg_pre, avg_rec, fscore


def _check(gt_labels, pred_labels):
    if gt_labels.ndim != 1:
        raise ValueError("gt_labels must be 1D: shape is %r" %
                         (gt_labels.shape,))
    if pred_labels.ndim != 1:
        raise ValueError("pred_labels must be 1D: shape is %r" %
                         (pred_labels.shape,))
    if gt_labels.shape != pred_labels.shape:
        raise ValueError(
            "gt_labels and pred_labels must have same size, got %d and %d" %
            (gt_labels.shape[0], pred_labels.shape[0]))
    return gt_labels, pred_labels


def getknn_brute(feature_path, labeltest, knnsave_path, n_neighbors):
    print("Creating knns graph...")
    featurearr = np.load(feature_path)
    # labelss = np.loadtxt(labeltest)
    labelss = np.load(labeltest)

    nbrs = KNeighborsClassifier(algorithm='brute', n_neighbors=n_neighbors).fit(featurearr, labelss)
    distances, indices = nbrs.kneighbors(featurearr)
    knnarr = np.array(indices)
    np.save(knnsave_path, knnarr)

    # distance_save_path= knnsave_path.strip('npy').split('/')[-1]
    distance_save_path=knnsave_path+'distance'
    np.save(distance_save_path, np.array(distances))


def labeltxt2npy(labeltxtfile, outputpath):
    temparr = []
    with open(labeltxtfile, 'r') as file:
        read = file.readlines()
        for lines in read:
            temparr.append(int(lines))
    conv = np.asarray(temparr)
    np.save(outputpath, conv)
