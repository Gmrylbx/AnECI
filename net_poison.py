"""
    Generating and seeding outliers.
    The details can found in paper.
"""

import numpy as np
import utils
import scipy.sparse as sp
import dataset
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=10,
                    help='Random seed.')
parser.add_argument('--dataset', type=str, default='polblogs', choices=['cora', 'citeseer', 'polblogs', 'pubmed'],
                    help='dataset')
parser.add_argument('--type', type=str, default='mix', choices=['s', 'a', 's&a', 'mix'],
                    help='pollution type')
parser.add_argument('--ratio', type=float, default=0.05,
                    help='pollution rate')
args = parser.parse_args()

utils.setup_seed(args.seed)

adj, features, labels, idx_train, idx_test, idx_val = dataset.load_datasp(args.dataset)
adj = np.array(adj.todense())
features = np.array(features.todense())
num_nodes = adj.shape[0]
num_ano = int(num_nodes * args.ratio)

if args.type == 's':
    ano_stru, ano_attr, ano_sa = num_ano, 0, 0
elif args.type == 'a':
    ano_stru, ano_attr, ano_sa = 0, num_ano, 0
elif args.type == 's&a':
    ano_stru, ano_attr, ano_sa = 0, 0, num_ano
elif args.type == 'mix':
    ano_stru = int(num_ano / 3)
    ano_attr = int(num_ano / 3)
    ano_sa = num_ano - ano_stru - ano_attr
else:
    exit("poison type error!")

# community members
community_members = []
class_nums = np.unique(labels)
for c in class_nums:
    community_members.append(np.where(labels == c)[0])
p = []
for c in class_nums:
    p.append(len(community_members[c]))
p = np.array(p)
p = p / p.sum()

# degree list
deg_list = []
for i in range(num_nodes):
    deg = sum(adj[i])
    deg_list.append(deg)
deg_list = np.array(deg_list)

max_deg_list = []
mean_deg_list = []
for c in class_nums:
    members = community_members[c]
    max_deg_list.append(deg_list[members].max())
    mean_deg_list.append(int(deg_list[members].mean()))

# poison graph
ano_nodes_classes = np.random.choice(class_nums, num_ano, p=p)
ano_adj = np.zeros([num_ano + num_nodes, num_ano + num_nodes])
ano_adj[0:num_nodes, 0:num_nodes] = adj
ano_feats = np.empty([num_ano, features.shape[1]])
labels = labels.tolist()
for i in range(num_ano):
    print(i)
    classes = ano_nodes_classes[i]
    labels.append(classes)
    edges = []
    attrs = []
    j, k, m = 0, 0, mean_deg_list[classes]
    if np.random.randint(0, 2, 1) == 0:
        m = int(1.1 * m)
    else:
        m = int(0.9 * m)

    if i < ano_stru:
        # structure anomaly
        while j < m:
            new_node = np.random.randint(0, num_nodes, 1)[0]
            if new_node not in community_members[classes] and new_node not in edges:
                edges.append(new_node)
                j += 1
        while k < m:
            new_node = np.random.randint(0, num_nodes, 1)[0]
            if new_node in community_members[classes] and new_node not in attrs:
                attrs.append(new_node)
                k += 1
        feats = features[attrs[np.random.randint(0, len(attrs), 1)[0]], :]
    elif i >= ano_stru + ano_attr:
        # combine anomaly
        new_class1, new_class2 = 0, 0
        if len(class_nums) == 2:
            while new_class2 == classes or new_class1 == classes:
                new_class1, new_class2 = np.random.randint(0, len(class_nums), 2)
        else:
            while new_class1 == new_class2 or new_class2 == classes or new_class1 == classes:
                new_class1, new_class2 = np.random.randint(0, len(class_nums), 2)
        while j < m:
            new_node = np.random.randint(0, num_nodes, 1)[0]
            if new_node in community_members[new_class1] and new_node not in edges:
                edges.append(new_node)
                j += 1
        while k < m:
            new_node = np.random.randint(0, num_nodes, 1)[0]
            if new_node in community_members[new_class2] and new_node not in attrs:
                attrs.append(new_node)
                k += 1
        feats = features[attrs[np.random.randint(0, len(attrs), 1)[0]], :]
    else:
        # attribute anomaly
        while j < m:
            new_node = np.random.randint(0, num_nodes, 1)[0]
            if new_node in community_members[classes] and new_node not in edges:
                edges.append(new_node)
                j += 1
        while k < m:
            new_node = np.random.randint(0, num_nodes, 1)[0]
            if new_node not in community_members[classes] and new_node not in attrs:
                attrs.append(new_node)
                k += 1
        feats = np.sum(features[attrs, :], axis=0) / len(attrs)

    # perturbations
    # bits = np.random.randint(0, len(feats), 2)
    # feats[bits] = 1 - feats[bits]

    ano_feats[i, :] = feats
    for h in range(len(edges)):
        ano_adj[num_nodes + i, edges[h]] = 1
        ano_adj[edges[h], num_nodes + i] = 1
labels = np.array(labels)
ano_feats = np.concatenate([features, ano_feats])
features = sp.csr_matrix(ano_feats)
adj = sp.csr_matrix(ano_adj)
anomaly_label = np.zeros(num_ano + num_nodes, dtype=np.int32)
anomaly_label[-num_ano:] = 1
