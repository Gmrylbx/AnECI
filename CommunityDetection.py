import time
import torch.optim as optim
import dataset
from model import AnECI
from utils import *
import args_communitydetection as args


def community_detection():
    # utils.setup_seed(args.seed)
    print("Using {} dataset".format(args.dataset))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load data
    adj, features, labels, idx_train, idx_test, idx_val = dataset.load_datasp(args.dataset)
    features = sp.eye(adj.shape[0], adj.shape[0])

    # preprocess data
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
    net.train()
    for epoch in range(1, args.num_epoch):
        time_start = time.time()
        opt.zero_grad()
        prob, embedding = net((features, support))
        I = torch.eye(info.shape[0]).to(device)
        Q = modularity(info - I * info, prob)
        R = reconstruct(prob, info, weight_tensor)
        loss = args.par1 * Q + args.par2 * R
        loss.backward()
        opt.step()

        time_end = time.time()
        time_epoch = time_end - time_start
        if args.print_yes and epoch % args.print_intv == 0:
            print("epoch %d :" % epoch, "time: %f" % time_epoch, "Q is %.10f" % (-1 * Q))

    net.eval()
    embedding = net.embedding((features, support))
    embedding = embedding.detach().to("cpu")
    sft = nn.Softmax(dim=1)
    prob = sft(embedding)
    modu = compute_Q(torch.Tensor(adj.todense()), prob)
    print("mudularity : %.4f" % modu)
    return modu


if __name__ == '__main__':
    modularity = community_detection()

