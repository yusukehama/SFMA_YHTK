### list of parameters 

# W matrix
num_K = 2 # number of columns of a matrix 
d_W = 10 # number of types of W matrices (instances) 

# inital data
nin_seed = 100 # the random seed for generating an initial dataset 

# Training FM model
seed_intial_FMTorch = 0 # random seed for generating FM model parameters (v,w,w0) for preparing augmented inital datasets 
LEARNING_RATE = 0.01 # the learning rate for training FM models
EPOCHS = 200 # number of epochs for training FM models

# simulated anneal
seed_SA = 0
nreads_SA = 10 # value of num_reads: the number of independent runs of SA 
nsweeps = 100 # number of Markov Chain Monte Carlo steps

# quanutm anneal
nreads_QA = 50 # value of num_reads: the number of independent runs of QA 

# SFMA
seed_sub  = 0 # random seed for generating a subsampling dataset

# rounded quantities
n_round = 17 # number of decimal places