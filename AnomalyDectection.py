import torch.optim as optim
from utils import *
from model import AnECI
import time
import args_anomalydection as args
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score


def anomaly_detection():
    # utils.setup_seed(args.seed)
    print("Using {} dataset".format(args.dataset))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load data
    root = './Anomaly_Data/'
    adj = sp.load_npz(root + args.dataset + '/' + args.type + '-adj.npz')
    features = sp.load_npz(root + args.dataset + '/' + args.type + '-features.npz')
    anomaly_label = np.load(root + args.dataset + '/' + args.type + '-anomaly_label.npy')
    labels = np.load(root + args.dataset + '/' + args.type + '-labels.npy')

    info = get_info(adj, args.field)
    info = info.to(device)

    features = sparse_to_tuple(features)
    supports = preprocess_adj(adj)

    adj_label = adj + sp.eye(adj.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    i = torch.from_numpy(adj_label[0]).long().to(device)
    v = torch.from_numpy(adj_label[1]).to(device)
    adj_label = torch.sparse.FloatTensor(i.t(), v, adj_label[2])
    weight_mask = adj_label.to_dense() == 1
    weight_tensor = torch.ones_like(weight_mask).to(device)
    weight_tensor[weight_mask] = 10
    stru_label = torch.zeros_like(info)
    stru_label[info > 0] = 1

    i = torch.from_numpy(features[0]).long().to(device)
    v = torch.from_numpy(features[1]).to(device)
    features = torch.sparse.FloatTensor(i.t(), v, features[2]).to(device)
    features = features.to(torch.float32)

    i = torch.from_numpy(supports[0]).long().to(device)
    v = torch.from_numpy(supports[1]).to(device)
    support = torch.sparse.FloatTensor(i.t(), v, supports[2]).float().to(device)

    num_features_nonzero = features._nnz()
    feat_dim = features.shape[1]

    net = AnECI(feat_dim, args.hidden[-1], args.hidden, args.dropout, num_features_nonzero)
    opt = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weightdecay)
    net.to(device)

    patience = args.patience
    step, time_total, best_score = 0, 0, 0
    for epoch in range(1, args.num_epoch):
        time_start = time.time()
        net.train()
        opt.zero_grad()
        prob, embedding = net((features, support))
        I = torch.eye(info.shape[0]).to(device)
        Q = modularity(info - I * info, prob)
        R = reconstruct(prob, info, weight_tensor)
        loss = args.par1 * Q + args.par2 * R
        loss.backward()
        opt.step()

        net.eval()
        sft = nn.Softmax(dim=1)
        embedding = net.embedding((features, support))
        embedding = embedding.detach().to("cpu")
        prob = sft(embedding)
        score = torch.sum(prob / torch.max(prob, dim=1, keepdim=True).values, dim=1).numpy()
        auc_score = roc_auc_score(anomaly_label, score)
        ap_score = average_precision_score(anomaly_label, score)
        time_end = time.time()
        time_epoch = time_end - time_start
        time_total += time_epoch
        if args.print_yes and epoch % args.print_intv == 0:
            print("epoch %d :" % epoch, "time: %f" % time_epoch, "Q is %.10f" % Q,
                  "auc : %.4f" % auc_score, "ap : %.4f" % ap_score)
        if best_score > Q:
            best_score = Q
            step = 0
        else:
            step += 1
        if step == patience and epoch > patience:
            break
    print("auc : %.4f" % auc_score, "ap : %.4f" % ap_score)
    return auc_score, ap_score


if __name__ == '__main__':
    auc, ap = anomaly_detection()

