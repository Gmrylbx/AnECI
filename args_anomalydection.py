################################### CONFIGS OF ANOMALY DETECTION ###################################

# optional arguments:
# --dataset  Dataset string.
# --type  types of anomaly node, including 's', 'a', 's&a' and 'mix', the details of outlier type can found in paper.
# --learning_rate  Initial learning rate.
# --num_epoch  Number of epochs to train.
# --hidden  Number of units in hidden layer.
# --dropout Dropout rate (1 - keep probability).
# --par1 Parameter for modularity loss, i.e. beta_1.
# --par2 Parameter for reconstruction loss, i.e. beta_2.
# --field Parameter of W_l, i.e. refers to the weights of calculating the high-order proximity matrix.


dataset = 'cora'
type = 's'
hidden = [128, 50]
dropout = [0.4, 0.4]
num_epoch = 800
learning_rate = 0.001
weightdecay = 0
par1, par2, = 1e3, 1e4
field = [1, 1e-1, 5e-3]
print_yes = 1
print_intv = 1
patience = 20


# dataset = 'citeseer'
# type = 'a'
# hidden = [128, 50]
# dropout = [0.4, 0.4]
# num_epoch = 800
# learning_rate = 0.001
# weightdecay = 0
# par1, par2, = 1e3, 1e4
# field = [1, 1e-1, 5e-3]
# print_yes = 1
# print_intv = 1
# patience = 20


# dataset = 'polblogs'
# type = 's&a'
# hidden = [512, 256]
# dropout = [0.4, 0.4]
# num_epoch = 800
# learning_rate = 0.001
# weightdecay = 0
# par1, par2, = 1e3, 1e4
# field = [1, 5, 1]
# print_yes = 1
# print_intv = 1
# patience = 40


# dataset = 'pubmed'
# type = 'mix'
# hidden = [128, 50]
# dropout = [0.4, 0.4]
# num_epoch = 800
# learning_rate = 0.001
# weightdecay = 0
# par1, par2, = 1e3, 1e4
# field = [1, 1e-1, 5e-3]
# print_yes = 1
# print_intv = 1
# patience = 40






