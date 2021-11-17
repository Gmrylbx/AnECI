################################### CONFIGS OF NODE CLASSIFICATION###################################

# optional arguments:
# --dataset  Dataset string.
# --attack  attack methods, including 'random', 'nettack' and 'FGA'
# --ptb_rate  number/rate of perturbations
# --learning_rate  Initial learning rate.
# --num_epoch  Number of epochs to train.
# --hidden  Number of units in hidden layer.
# --dropout Dropout rate (1 - keep probability).
# --par1 Parameter for modularity loss, i.e. beta_1.
# --par2 Parameter for reconstruction loss, i.e. beta_2.
# --field Parameter of W_l, i.e. refers to the weights of calculating the high-order proximity matrix.
# --a Parameter of smoothing function in AnECI+, i.e. Alpha, and we fix Beta, Gama as 0.5, 0.75, respectively.
# --seed random seed


########################### Parameter setting on clean datasets ###########################
denoise = 0

dataset = 'cora'
attack = 'no'
ptb_rate = 0
hidden = [1024, 512]
dropout = [0.4, 0.4]
num_epoch = 150
learning_rate = 0.0001
weightdecay = 0
par1, par2 = 1e3, 1e4         ### beta_1, beta_2
field = [1, 1e-1, 1e-3]       ### W_l
print_yes = 1
print_intv = 1

# dataset = 'citeseer'
# attack = 'no'
# ptb_rate = 0
# hidden = [1024, 512]
# dropout = [0.4, 0.4]
# num_epoch = 150
# learning_rate = 0.0001
# weightdecay = 5e-4
# par1, par2 = 1e4, 1e4
# field = [1, 1e-1, 1e-3]
# print_yes = 1
# print_intv = 1

# dataset = 'polblogs'
# attack = 'no'
# ptb_rate = 0
# hidden = [1024, 512]
# dropout = [0.4, 0.4]
# num_epoch = 150
# learning_rate = 0.0001
# weightdecay = 0
# par1, par2 = 1e4, 1e4
# field = [1, 1e-1, 1e-3]
# print_yes = 1
# print_intv = 1

# dataset = 'pubmed'
# attack = 'no'
# ptb_rate = 0
# hidden = [512]
# dropout = [0.4]
# num_epoch = 150
# learning_rate = 0.01
# weightdecay = 0
# par1, par2 = 1e3, 1e4
# field = [1]
# print_yes = 1
# print_intv = 1




########################### Parameter setting on attacked datasets ###########################
# # # # # # # # # # #
#   nettack         #
#   cora : a=5      #
#   citeseer : a=5  #
#   polblogs : a=5  #
#   pubmed : a=12   #
# # # # # # # # # # #

# # # # # # # # # # #
#   FGA             #
#   cora : a=4      #
#   citeseer : a=2  #
#   polblogs : a=18 #
#   pubmed : a=3    #
# # # # # # # # # # #

# # # # # # # # # # #
#   random          #
#   cora : a=2      #
#   citeseer : a=2  #
#   polblogs : a=12 #
#   pubmed : a=4    #
# # # # # # # # # # #


# denoise = 1
# dataset = 'cora'
# attack = 'nettack'
# ptb_rate = 5
# a = 5
# hidden = [1024, 512]
# dropout = [0.4, 0.4]
# num_epoch = 150
# learning_rate = 0.0001
# weightdecay = 0
# par1, par2 = 1e3, 1e4
# field = [1, 1e-1]
# print_yes = 1
# print_intv = 1


# denoise = 1
# dataset = 'citeseer'
# attack = 'FGA'
# ptb_rate = 1
# a = 2
# hidden = [1024, 512]
# dropout = [0.4, 0.4]
# num_epoch = 150
# learning_rate = 0.0001
# weightdecay = 5e-4
# par1, par2 = 1e4, 1e4
# field = [1, 1e-1]
# print_yes = 1
# print_intv = 50


# denoise = 1
# dataset = 'polblogs'
# attack = 'random'
# ptb_rate = 0.5
# a = 12
# hidden = [1024, 512]
# dropout = [0.4, 0.4]
# num_epoch = 150
# learning_rate = 0.0001
# par1, par2 = 1e4, 1e4
# field = [1, 1e-1]
# print_yes = 1
# print_intv = 50


# denoise = 1
# dataset = 'pubmed'
# attack = 'nettack'
# a = 12
# ptb_rate = 5
# hidden = [512]
# dropout = [0.4]
# num_epoch = 150
# learning_rate = 0.01
# weightdecay = 0
# par1, par2 = 1e3, 1e4
# field = [1]
# print_yes = 1
# print_intv = 50

