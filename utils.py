import torch
import random
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score as acc
from sklearn.multiclass import OneVsRestClassifier
import scipy.sparse as sp
import json


def get_prognn_splits(json_file):
    """Get target nodes incides, which is the nodes with degree > 10 in the test set."""
    with open(json_file, 'r') as f:
        idx = json.loads(f.read())
    return np.array(idx['idx_train']), \
           np.array(idx['idx_val']), np.array(idx['idx_test'])


def check_classification(x_train, x_test, y_train, y_test):
    clf = OneVsRestClassifier(LogisticRegression(max_iter=10000, penalty='l2'))
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    ac = acc(y_test, y_pred)
    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")
    return ac, micro, macro


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


def sparse_dropout(x, rate, noise_shape):
    """

    :param x:
    :param rate:
    :param noise_shape: int scalar
    :return:
    """
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).bool()
    i = x._indices() # [2, 49216]
    v = x._values() # [49216]

    # [2, 4926] => [49216, 2] => [remained node, 2] => [2, remained node]
    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)

    out = out * (1. / (1-rate))
    return out


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^-0.5AD^-0.5


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def get_info(adj, field):
    current = adj
    stru = field[0] * adj
    for i in range(1, len(field)):
        current = current.dot(adj)
        stru += field[i] * current
    rowsum = np.array(stru.sum(1), dtype=np.float32)  # get sum of each row, [2708, 1]
    r_inv = np.power(rowsum, -1).flatten()  # 1/rowsum, [2708]
    r_inv[np.isinf(r_inv)] = 0.  # zero inf data
    r_mat_inv = sp.diags(r_inv)  # sparse diagonal matrix, [2708, 2708]
    stru = r_mat_inv.dot(stru)
    info = torch.Tensor(stru.todense())
    return info


def modularity(stru, prob):
    m = torch.sum(stru) / 2
    B = stru - (torch.sum(stru, dim=1, keepdim=True) *
                torch.transpose(torch.sum(stru, dim=1, keepdim=True), dim0=0, dim1=1)) / (2 * m)
    Q = torch.trace(torch.mm(torch.mm(torch.transpose(prob, dim0=0, dim1=1), B), prob)) / (2 * m)
    return -1 * Q


def reconstruct(prob, stru, weight_tensor):
    b_xent = nn.BCEWithLogitsLoss(weight=weight_tensor)
    R = b_xent(torch.matmul(prob, prob.t()), stru)
    return R


def compute_Q(adj, prob):
    comm_labels = prob.argmax(dim=1).numpy()
    comm_dict = {}
    comm_index = 0
    for i in range(len(comm_labels)):
        if comm_labels[i] not in comm_dict:
            comm_dict[comm_labels[i]] = comm_index
            comm_index += 1
    comm_onehot = torch.zeros([len(comm_labels), len(np.unique(comm_labels))])
    for i in range(len(comm_labels)):
        comm_onehot[i][comm_dict[comm_labels[i]]] = 1
    Q = modularity(adj, comm_onehot)
    return -1 * Q


def topL(ano_labels, score):
    s_sorted = np.sort(score)[::-1]
    s_sorted_index = np.argsort(score)[::-1]
    # print(score[s_sorted_index[1000]] == s_sorted[1000])
    ano_labels = ano_labels[s_sorted_index]
    recall = []
    for i in range(1, 6):
        ratio = i * 0.05
        num = int(len(ano_labels) * ratio)
        label_pred = ano_labels[0:num]
        recall.append(sum(label_pred) / sum(ano_labels))
    return recall
