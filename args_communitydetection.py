################################### CONFIGS OF COMMUNITY DETECTION ###################################

# optional arguments:
# --dataset  Dataset string.
# --learning_rate  Initial learning rate.
# --num_epoch  Number of epochs to train.
# --hidden  Number of units in hidden layer.
# --dropout Dropout rate (1 - keep probability).
# --par1 Parameter for modularity loss, i.e. beta_1.
# --par2 Parameter for reconstruction loss, i.e. beta_2.
# --field Parameter of W_l, i.e. refers to the weights of calculating the high-order proximity matrix.


dataset = 'citeseer'  # 'citeseer', 'polblogs', 'pubmed'
hidden = [128]
dropout = [0.4]
num_epoch = 600
learning_rate = 0.01
weightdecay = 0
par1, par2 = 1e4, 1e4
field = [1, 1e-1]
print_yes = 1
print_intv = 100