import time
import torch.optim as optim
from deeprobust.graph.data import PrePtbDataset
import dataset
from utils import *
from model import AnECI
import args_nodeclassification as args


def train(data, par, model, opt, device):
    # data
    adj, features, labels = data['adj'], data['features'], data['labels']
    idx_train, idx_val, idx_test = data['idx_train'], data['idx_val'], data['idx_test']

    # parameters
    epochs, par1, par2, field = par['epochs'], par['par1'], par['par2'], par['field']
    print_yes, print_intv = par['print_yes'], par['print_intv']

    # data preprocess
    y_train = labels[idx_train]
    y_val = labels[idx_val]
    y_test = labels[idx_test]

    info = get_info(adj, field)   # High-order proximity matrix
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

    model.to(device)
    time_total, best_score, best_epoch = 0, 0, -1
    for epoch in range(1, epochs + 1):
        time_start = time.time()
        model.train()
        prob, embedding = model((features, support))
        I = torch.eye(info.shape[0]).to(device)
        Q = modularity(info - I * info, prob)
        R = reconstruct(prob, info, weight_tensor)
        loss = par1 * Q + par2 * R
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        embedding = model.embedding((features, support))
        embedding = embedding.detach().to("cpu")
        embedding = embedding.numpy()

        x_train, x_val, x_test = embedding[idx_train, :], embedding[idx_val, :], embedding[idx_test, :]
        acc_val, micro_val, macro_val = check_classification(x_train, x_val, y_train, y_val)
        if acc_val > best_score:
            best_epoch = epoch
            best_embedding = embedding
            best_score = acc_val

        time_end = time.time()
        time_epoch = time_end - time_start
        time_total += time_epoch
        if print_yes and epoch % print_intv == 0:
            print("epoch %d :" % epoch, "time: %.2f" % time_epoch, "Q : %.10f" % (-1 * Q), "acc_val : %.4f" % acc_val)

    print("best epoch : %d, " % best_epoch, "best val score : %.4f" % best_score)
    return best_score, best_embedding, best_epoch


def node_classfication():
    setup_seed(5)
    print("Using {} dataset".format(args.dataset))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load clean data set
    adj, features, labels, idx_train, idx_test, idx_val = dataset.load_datasp(args.dataset)

    # load attacked data set
    if args.ptb_rate == 0:
        args.attack = 'no'
    if args.attack == 'random':
        print("random attack with rate %.2f" % args.ptb_rate)
        from deeprobust.graph.global_attack import Random
        # to fix the seed of generated random attack, you need to fix both np.random and random
        # you can uncomment the following code
        # import random
        # random.seed(args.seed)
        # np.random.seed(args.seed)
        attacker = Random()
        n_perturbations = int(args.ptb_rate * (adj.sum() // 2))
        attacker.attack(adj, n_perturbations, type='add')
        adj = attacker.modified_adj
    if args.attack == 'nettack':
        perturbed_data = PrePtbDataset(root='./Nettack_Data/',
                                       name=args.dataset,
                                       attack_method=args.attack,
                                       ptb_rate=args.ptb_rate)
        adj = perturbed_data.adj
        idx_test = perturbed_data.target_nodes
    if args.attack == 'FGA':
        root = './FGA_Data/'
        print('Reading structure from', root)
        adj = sp.load_npz(root + args.dataset + '_FGA_adj_' + str(args.ptb_rate) + '.npz')
        json_file = root + args.dataset + '_FGA_nodes.json'
        with open(json_file, 'r') as f:
            idx = json.loads(f.read())
        idx_test = idx["attacked_test_nodes"]
        idx_test = np.array(idx_test)

    y_train = labels[idx_train]
    y_val = labels[idx_val]
    y_test = labels[idx_test]

    data, par = {}, {}
    # data
    data['adj'], data['features'], data['labels'] = adj, features, labels
    data['idx_train'], data['idx_val'], data['idx_test'] = idx_train, idx_val, idx_test
    adj, features, labels = data['adj'], data['features'], data['labels']
    num_features_nonzero, feat_dim, node_dim = features.nnz, features.shape[1], adj.shape[1]
    # parameters
    par['epochs'], par['par1'], par['par2'], par['field'] = args.num_epoch, args.par1, args.par2, args.field
    par['print_yes'], par['print_intv'] = args.print_yes, args.print_intv

    adj_drop = adj
    if args.denoise:
        print("***********************************************denoising phase start***********************************************")
        net_denoise = AnECI(feat_dim, args.hidden[-1], args.hidden, args.dropout, num_features_nonzero)
        opt = optim.Adam(net_denoise.parameters(), lr=args.learning_rate, weight_decay=args.weightdecay)
        best_score, best_embedding, best_epoch = train(data, par, net_denoise, opt, device)
        embedding = torch.Tensor(best_embedding)

        ######## calculating anomaly score of edges ########
        norm = torch.norm(embedding, p=2, dim=1, keepdim=True)
        embedding = torch.div(embedding, norm)
        adj_rec = torch.mm(embedding, embedding.t())
        adj = torch.Tensor(adj.todense())
        adj_rec = adj - adj_rec * adj
        adj_rec_1d = adj_rec.view(-1)
        adj_rec_1d = adj_rec_1d[adj_rec_1d > 0]
        induces = torch.flip(torch.sort(adj_rec_1d).indices, dims=[0])
        adj_rec_1d = adj_rec_1d[induces]
        ######## calculating anomaly score of edges ########

        ratio = adj_rec.sum() / (adj).sum()  ###drop ratio
        a, b, c = args.a, 0.5, 0.75
        ratio = 1 / (1 + np.exp((b - ratio) * a)) * c
        print("drop ratio : %.4f" % (ratio))

        thre = adj_rec_1d[int(ratio * len(adj_rec_1d))]
        adj_drop = adj_rec.clone()
        adj_drop[adj_drop >= thre] = 1
        adj_drop[adj_drop < thre] = 0
        adj_drop = adj - adj_drop
        adj_drop = sp.csr_matrix(adj_drop.numpy())
        print("***********************************************denoising phase over***********************************************\n")


    print("***********************************************training phase start***********************************************")
    data['adj'] = adj_drop
    par['epochs'], par['par1'], par['par2'], par['field'] = args.num_epoch, args.par1, args.par2, args.field
    par['print_yes'], par['print_intv'], par['fastmode'] = 1, args.print_intv, 0

    net = AnECI(feat_dim, args.hidden[-1], args.hidden, args.dropout, num_features_nonzero)
    opt = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weightdecay)
    acc_val, embedding, best_epoch = train(data, par, net, opt, device)

    x_train = embedding[idx_train, :]
    x_test = embedding[idx_test, :]

    acc_test, micro_test, macro_test = check_classification(x_train, x_test, y_train, y_test)
    print("***********************************************training phase over***********************************************")

    print("\ntrain over!   best_epoch : %d,  acc_test: %.4f\n" % (best_epoch, acc_test))
    return acc_test, micro_test, macro_test


if __name__ == '__main__':
    acc, micro, macro = node_classfication()