import sys, os, datetime, time, pickle, bz2, itertools, glob, random, neal, dimod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from dimod import BinaryQuadraticModel as BQM
from pprint import pformat
from IPython.display import clear_output
from typing import Callable, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from tqdm import trange
from dwave.system import DWaveSampler, EmbeddingComposite

from parameters_SFMA import *


### Basic functions

def binary_array_to_decimal(x: np.ndarray) -> int:
    """
    Convert a binary vector (NumPy array) to its decimal (base-10) representation (int).

    Parameters:
        x (np.ndarray): Binary vector.

    Returns:
        int: Base-10 integer representing an index of x in a binary solution space.
    """
    
    binary_str = ''.join(map(str, x)) # string of binaries
    return int(binary_str, 2) # conversion

def calculate_binary_quadratic(x: np.ndarray, A: np.ndarray, b: float) -> float:
    """
    Calculate an objective function (float) in a quadratic form of a binary vector (NumPy array): QUBO function or Factorization Machine   
    (FM) function.

    Parameters:
        x (np.ndarray): Binary vector.
        A (np.ndarray): Matrix.
        b (float): Constant.

    Returns:
        float: Objective function.
    """
    
    return np.dot(x.T, np.dot(A, x)) + b

def blackbox(W: np.ndarray, x: np.ndarray, n_N: int, n_K: int) -> float:
    """
    Calculate a black-box function (float). A binary vector x (NumPy array) is converted to a n_N (int) by n_K (int) matrix M (NumPy array).  

    Parameters:
        W (np.ndarray): W matrix.
        x (np.ndarray): Binary vector.
        n_N (int): Number of rows of M.
        n_K (int): Number of columns of M.

    Returns:
        float: Black-box function representing the Frobenius norm of the difference between W (NumPy array) and its approximated matrix V(M, W) (NumPy array):    
        V(M, W) = M · (M^T · M)^{-1} · M^T · W.   
    """
    x_spin = 2*x - 1 # convert a binary vector to a spin vector
    M = x_spin.reshape((n_N, n_K)) # construct a M matrix
    C = np.linalg.pinv(M.T @ M) @ M.T @ W # preparation for formulating V(M, W)
    V = M @ C # V(M, W) 
    e = np.sum((W - V)**2) # black-box function
    return e  


def all_binaries(d: int) -> np.ndarray:
    """
    Generate all binary vectors (NumPy array) of dimension d (int).

    Parameters:
        d (int): Number of spins (qubits).

    Returns:
        np.ndarray: All d-dimensional binary vectors.
    """
    
    return np.array(list(itertools.product([0, 1], repeat=d)))

 
def init_training_data(W: np.ndarray, n_N: int, n_K: int, d: int, n_0: int, n_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate an intial dataset of n_0 (int) samples described as a tuple (x, y) by randomly selecting n_0 binary vectors with a random seed n_seed (int).
    x (np.ndarray): Input data (d-dimensional binary vectors).
    y (np.ndarray): Output data (corresponidng values of a black-box function).
    
    Parameters:
        W (np.ndarray): W matrix.
        n_N (int): Number of rows of M.
        n_K (int): Number of columns of M.
        d (int): Number of spins (qubits).
        n_0 (int): Size of an inital dataset.
        n_seed (int): Random seed.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Initial dataset (x,y).
    """
    
    assert n_0 < 2**d  # ensure enough unique samples exist
    
    all_vectors = all_binaries(d) # return all binary vectors
    
    # Sample n_0 unique data samples.
    rg = np.random.default_rng(n_seed) # random number generator
    x = all_vectors[rg.choice(2**d, size=n_0, replace=False)] # inputs
    y = np.array([blackbox(W, xi, n_N, n_K) for xi in x])  # outputs
    return x, y

def standard(y: np.ndarray, Mean: float, Stand_Dev: float) -> np.ndarray:
    """
    Calculate a standardized output data vector (NumPy array) given by Mean (float) and Stand_Dev (float).
    
    Parameters:
        y (np.ndarray): Outputs of a dataset.
        Mean (float): Mean calculated by randomly selecting data points (retrieved from '20241227_pFMA_Wmatrix/results/z900/').
        Stand_Dev (float): Standard deviation calculated by randomly selecting data points (retrieved from '20241227_pFMA_Wmatrix/results/z900/').

    Returns:
        np.ndarray: Standardized output vector.
    """
    
    return (y - Mean) / Stand_Dev

### Introduce a FM class and a training function.


class TorchFM(nn.Module):
    def __init__(self, d: int, k: int, Stand_Dev: float, v_init: torch.Tensor =None, w_init: torch.Tensor =None, w0_init: torch.Tensor =None): 
        """
        Introduce an FM class: TorchFM
        
        Parameters:
            d (int): Number of spins (qubits).
            k (int): Hyperparameter for an FM model.
            Stand_Dev (float): Standard deviation. 
            v_init (torch.Tensor): Initial values for v (d by k matrix). 
            w_init (torch.Tensor): Initial values for w (d-dimensional vector).
            w0_init (torch.Tensor): Initial value for w0 (scalar). 

        Returns:
            TorchFM: FM model instance.    
        """
       
        super().__init__()
        self.d = d

        Std_Torch = torch.tensor(Stand_Dev, dtype=torch.float64) # convert Stand_Dev (float) to a PyTorch tensor                                                        
        # Initialize v.
        if v_init is None:
            self.v = nn.Parameter(torch.randn((d, k), dtype=torch.float64) * Std_Torch, requires_grad=True) # inital v as a random tensor 
        else:
            self.v = nn.Parameter(v_init.clone().detach(), requires_grad=True)  # previously trained (updated) v
        
        # Initialize w.
        if w_init is None:
            self.w = nn.Parameter(torch.randn((d,), dtype=torch.float64) * Std_Torch, requires_grad=True) # inital w as a random vector 
        else:
            self.w = nn.Parameter(w_init.clone().detach(), requires_grad=True)  # previously trained w
        
        # Initialize w0.
        if w0_init is None:
            self.w0 = nn.Parameter(torch.randn((), dtype=torch.float64) * Std_Torch, requires_grad=True) # inital w0 as a random number
        else:
            self.w0 = nn.Parameter(w0_init.clone().detach(), requires_grad=True)  # previously trained w0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to calculate the predicted values of an FM model or an FM function (Pytorch tensor).

        Parameters:
            x (torch.Tensor): Input data vectors (d-dimensional binary vectors).

        Returns:
            torch.Tensor: Predicted values of an FM function.
        """
        
        # Formulate a binary quadratic model: FM model.
        
        out_linear = torch.matmul(x, self.w) + self.w0 # linear term

        # Quadratic term
        out_1 = torch.matmul(x, self.v).pow(2).sum(1)
        out_2 = torch.matmul(x.pow(2), self.v.pow(2)).sum(1)
        out_quadratic = 0.5 * (out_1 - out_2) # finalized form

        # Combine linear and quadratic terms.
        out = out_linear + out_quadratic
        return out
    
    def get_parameters(self) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Get FM model parameters v (NumPy array), w (NumPy array), and w0 (float).

        Returns:
            tuple[np.ndarray, np.ndarray, float]: FM model parameters (v, w, w0).
        """
        
        # Detach v, w, and w0, convert them to NumPy arrays and float, and copy them. 
        np_v = self.v.detach().numpy().copy() # v
        np_w = self.w.detach().numpy().copy() # w
        np_w0 = self.w0.detach().numpy().copy() # w0
        return np_v, np_w, float(np_w0)


def train_FM_model(x: np.ndarray, y: np.ndarray, model: TorchFM) -> Tuple[float, Tuple[np.ndarray, np.ndarray, float]]:
    """
    Train an FM model (TorchFM) using full-batch optimization.

    Parameters:
        x (np.ndarray): Inputs of a training dataset.
        y (np.ndarray): Outputs of a training dataset. 
        model (TorchFM): FM model instance to be trained.

    Returns:
        Tuple[float, Tuple[np.ndarray, np.ndarray, float]]:
        A (float): Final value of a loss function.
        B (np.ndarray): Trained FM model parameter v.
        C (np.ndarray): Trained FM model parameter w.
        D (float): Trained FM model parameter w0.        
    """
    
    # Convert an inital dataset (NumPy arrays) to PyTorch tensors.
    x_tensor = torch.from_numpy(x).double() # inputs 
    y_tensor = torch.from_numpy(y).double() # outputs

    # Full-batch optimization
    batch_size = x_tensor.shape[0]  # batch size
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # optimizer: Adam 
    loss_func = nn.MSELoss() # loss function 

    # Train (v,w,w0).
    for epoch in range(EPOCHS):
        optimizer.zero_grad() # reset all the previous gradients 
        
        # Forward pass
        pred_y = model(x_tensor) # compute predicted values
        loss = loss_func(pred_y, y_tensor) # compute a loss function
        
        # Backward pass and parameter update     
        loss.backward() # backpropagation 
        optimizer.step() # update training parameters
    
    # Extract the final FM parameters.
    trained_v, trained_w, trained_w0 = model.get_parameters() # updated v, w, and w0
    
    return loss.item(), (trained_v, trained_w, trained_w0) # updated loss function and (v, w, w0)

### Introduce a simulated annealer (SA).

def construct_QUBO(w: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Construct a QUBO matrix (NumPy array) in terms of FM model parameters w (NumPy array) and v (NumPy array).

    Parameters:
        w (np.ndarray): FM model parameters for diagonal elements of a QUBO matrix.
        v (np.ndarray): FM model parameters for off-diagonal elements of a QUBO matrix.

    Returns:
        np.ndarray: QUBO matrix.
    """
    
    dim = len(w) # dimension of a binary vector: the number of spins (qubits)
    
    # Construct diagonal elements. 
    Q_diag=np.zeros((dim,dim)) # zero matrix
    for i in range(dim):
        Q_diag[i,i] = w[i] # set the (i,i)th element of a QUBO matrix to the ith component of w

    # Construct off-diagonal elements. 
    Q_off_diag=np.zeros((dim,dim)) # zero matrix
    for i in range(dim):
        for j in range(i+1,dim):
            vi = v[i] # ith component of v
            vj = v[j] # jth component of v
            Q_off_diag[i,j] = np.dot(vi,vj) # set the (i,j)th element of a QUBO matrix to the inner product (vi,vj)
    Q_qubo = Q_diag+Q_off_diag # QUBO matrix
    return Q_qubo

def sample_to_vector(sample: dict, d: int) -> np.ndarray:
    """
    Convert a sample (dictionary of binaries) to a binary vector ordered by indices from 0 to d-1 (d: int).

    Parameters:
        sample (dict): Dictionary obtained from an annealer, where keys are spin (qubit) indices and values are their corresponding binaries.  
        d (int): Number of spins (qubits).

    Returns:
        np.ndarray: Binary vector ordered by indices from 0 to d-1.
    """

    variable_order = list(range(d))
    return np.array([sample[v] for v in variable_order])


sa_sampler = neal.SimulatedAnnealingSampler() # SA sampler

def simulated_anneal(trained_FMmodel_set: Tuple[float, Tuple[np.ndarray, np.ndarray, float]], W: np.ndarray, n_N: int, n_K: int
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    SA for acquiring both candidate solutions or samples (NumPy array) and the corresponding objective values or energies (NumPy array) 
    and the best (optimal) solution (NumPy array) and the corresponding value (float).

    Parameters:
        Tuple[float, Tuple[np.ndarray, np.ndarray, float]]: (loss function, FM model parameters (v, w, w0)).
        W (np.ndarray): W matrix.
        n_N (int): Number of rows of M.
        n_K (int): Number of columns of M.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        np.ndarray: Candidate solutions.
        np.ndarray: Objective values of the candidate solutions.
        np.ndarray: Best (optimal) solution.
        float: Objective value of the best (optimal) solution.
    """

    # Acquire parameters v, w, w0 from a FM model instance.
    _, trained_parameters = trained_FMmodel_set # (v,w,w0): drop a loss function
    v, w, w0 = trained_parameters # v, w, and w0

    # Construct a FM model.
    Q_qubo = construct_QUBO(w, v) # QUBO matrix (FM function)

    # Program SA.
    bqm = BQM.from_qubo(Q_qubo) # convert Q_qubo to BQM
    res_sa = sa_sampler.sample(bqm, num_reads=nreads_SA, num_sweeps=nsweeps, seed=seed_SA) # run SA
    
    # Extract the candidate solutions and corresponding energies.
    response_x = res_sa.record['sample'] # samples
    response_y = res_sa.record['energy'] # energies
    nbit = n_N*n_K # number of spins (qubits)
    xhat = sample_to_vector(res_sa.first.sample, nbit) # Best solution ordered by indices from 0 to nbit-1.
    
    # Check if a valid solution was returned.
    if len(xhat) == 0:
        raise RuntimeError("No solution was found.")
    yhat = blackbox(W, xhat, n_N, n_K) # corresponding objective value
    return  response_x, response_y, xhat, yhat

### Introduce a function for generating a sampling subdataset and training functions with a fulldataset and a subdataset via SA.

def uniform_int(n_fullD: int, n_subD: int, seed: int) -> np.ndarray:
    """
    Array of n_subD integers (n_subD: int) in the range [0,n_fullD-1] (n_fullD: int) sampled via a uniform distribution with a fixed random seed (seed: int).

    Parameters:
        n_fullD (int): Size of a full dataset.
        n_subD (int): Size of a subsampling dataset.
        seed (int): Random seed.

    Returns:
        np.ndarray: Array of n_subD integers representing indices of input binary vectors used for running SFMA.        
    """
    
    rng_uni = np.random.default_rng(seed) # random number generator
    uniform = rng_uni.integers(0, n_fullD, n_subD) # a generator of n_subD random integers 
    return uniform

def subdata_uniform(x: np.ndarray, y: np.ndarray, R: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generating a subdataset (tuple (x_random, y_random)) given by a ratio R (float).

    Parameters:
        x (np.ndarray): Inputs of a dataset.
        y (np.ndarray): Outputs of a dataset.
        R (float): Ratio.
        seed (int): Random seed.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Subsampling dataset.
    """
    
    dim = len(y) # dimension of a full dataset 
    eff_dim= int(dim*R) # dimension of a subdataset 
    randomN_array = uniform_int(dim, eff_dim, seed) # a generator of eff_dim random integers: indices of the elements of the subdataset
    xlist = [] # list for inputs 
    ylist = [] # list for outputs
    for a in randomN_array:
        xlist.append(x[a]) # append the ath input
        ylist.append(y[a]) # append the ath output
    x_random, y_random = np.array(xlist), np.array(ylist) # inputs and outputs of the subdataset
    return x_random, y_random

### Training functions with a fulldataset and a subdataset via SA.
    
def train_full_standard_SA(x: np.ndarray, y: np.ndarray, W: np.ndarray, n_N: int, n_K: int, Mean: float, Stand_Deviate: float, seed_Torch: int, num_it: int
                          ) -> Tuple[Tuple[float, Tuple[np.ndarray, np.ndarray, float]], np.ndarray, np.ndarray]:
    """
    Run standardized FMA via SA.

    Parameters:
        x (np.ndarray): Initial inputs of a dataset.
        y (np.ndarray): Initial outputs of a dataset.
        W (np.ndarray): W matrix.
        n_N (int): Number of rows of M.
        n_K (int): Number of columns of M.
        Mean (float): Mean.
        Stand_Deviate (float): Standard deviation.
        seed_Torch (int): Random seed.
        num_it (int): Number of the iterations of FMA. 
        
    Returns: 
        Tuple[Tuple[float, Tuple[np.ndarray, np.ndarray, float]], np.ndarray, np.ndarray]:
        A (float): Final loss function. 
        B (np.ndarray): Final FM model parameter v.
        C (np.ndarray): Final FM model parameter w.
        D (float): Final FM model parameter w0.
        E (np.ndarray): Final input data x.
        F (np.ndarray): Final output data y.
    """

    Nbit = n_N*n_K # number of spins (qubits)
    kFM = int(Nbit/2)-1 # hyperparameter for a FM model 
    xs, ys = x, y # dataset to be updated
    for i in trange(num_it): # FMA loop
        y_stand = standard(ys, Mean, Stand_Deviate) # standardization
        # Train the model.
        torch.manual_seed(seed_Torch) # set a random seed for reproducibility 
        model = TorchFM(d=Nbit, k=kFM, Stand_Dev=Stand_Deviate) # instantiate a FM model 
        trained_FM_set = train_FM_model(xs, y_stand, model) # train a FM model
        # Perform annealing, get a new data point via SA, and augment a dataset.
        _, _, xast, yast = simulated_anneal(trained_FM_set, W, n_N, n_K) # SA
        # Augment the dataset with a new data point.
        xs = np.vstack((xs, xast)) # augmented x 
        ys = np.append(ys, yast) # augmented y
        
    return trained_FM_set, xs, ys


def train_sub_standard_SA(x: np.ndarray, y: np.ndarray, W: np.ndarray, n_N: int, n_K: int, Mean: float, Stand_Deviate: float, seed_Torch: int, num_it: int, R: float, num_seed: int
                         ) -> Tuple[Tuple[float, Tuple[np.ndarray, np.ndarray, float]], np.ndarray, np.ndarray]:
    """
    Run standardized SFMA via SA.

    Parameters:
        x (np.ndarray): Initial inputs of a dataset.
        y (np.ndarray): Initial outputs of a dataset.
        W (np.ndarray): W matrix.
        n_N (int): Number of rows of M.
        n_K (int): Number of columns of M.
        Mean (float): Mean.
        Stand_Deviate (float): Standard deviation.
        seed_Torch (int): Random seed.
        num_it (int): Number of the iterations of SFMA.
        R (float): Ratio.
        num_seed (int): Random seed.
 
    Returns: 
        Tuple[Tuple[float, Tuple[np.ndarray, np.ndarray, float]], np.ndarray, np.ndarray]:
        A (float): Final loss function. 
        B (np.ndarray): Final FM model parameter v.
        C (np.ndarray): Final FM model parameter w.
        D (float): Final FM model parameter w0.
        E (np.ndarray): Final input data x.
        F (np.ndarray): Final output data y.
    """  

    Nbit = n_N*n_K # number of spins (qubits)
    kFM = int(Nbit/2)-1 # hyperparameter for a FM model
    xs, ys = x, y # dataset to be updated
    x_sub, y_sub = subdata_uniform(xs, ys, R, num_seed) # initial subdataset
    for i in trange(num_it): # SFMA loop
        y_sub_stand = standard(y_sub, Mean, Stand_Deviate) # standardization
        # Train the FM model again with the new subdataset.
        torch.manual_seed(seed_Torch) # set a random seed for reproducibility 
        model = TorchFM(d=Nbit, k=kFM, Stand_Dev=Stand_Deviate) # instantiate a FM model
        trained_FM_set = train_FM_model(x_sub, y_sub_stand, model) # train a FM model
        # Perform annealing, get a new data point via SA, and augment a dataset.
        _, _, xast, yast = simulated_anneal(trained_FM_set, W, n_N, n_K) # SA
        # Augment the dataset with a new data point and create a new subdataset.
        xs = np.vstack((xs, xast)) # augmented x
        ys = np.append(ys, yast) # augmented y
        x_sub, y_sub = subdata_uniform(xs, ys, R, num_seed) # new subdataset 
    return trained_FM_set, xs, ys

def train_full_nonstandard_SA(x: np.ndarray, y: np.ndarray, W: np.ndarray, n_N: int, n_K: int, Stand_Deviate: float, seed_Torch: int, num_it: int
                             ) -> Tuple[Tuple[float, Tuple[np.ndarray, np.ndarray, float]], np.ndarray, np.ndarray]:
    """
    Run non-standardized FMA via SA.

    Parameters:
        x (np.ndarray): Initial inputs of a dataset.
        y (np.ndarray): Initial outputs of a dataset.
        W (np.ndarray): W matrix.
        n_N (int): Number of rows of M.
        n_K (int): Number of columns of M.
        Stand_Deviate (float): Standard deviation.
        seed_Torch (int): Random seed.
        num_it (int): Number of the iterations of FMA.
        
    Returns: 
        Tuple[Tuple[float, Tuple[np.ndarray, np.ndarray, float]], np.ndarray, np.ndarray]:
        A (float): Final loss function. 
        B (np.ndarray): Final FM model parameter v.
        C (np.ndarray): Final FM model parameter w.
        D (float): Final FM model parameter w0.
        E (np.ndarray): Final input data x.
        F (np.ndarray): Final output data y.
    """
    
    Nbit = n_N*n_K # number of spins (qubits)
    kFM = int(Nbit/2)-1 # hyperparameter for a FM model
    xs, ys = x, y # dataset to be updated
    for i in trange(num_it): # FMA loop
        # Train the model.
        torch.manual_seed(seed_Torch) # set a random seed for reproducibility 
        model = TorchFM(d=Nbit, k=kFM, Stand_Dev=Stand_Deviate) # instantiate a FM model
        trained_FM_set = train_FM_model(xs, ys, model) # train a FM model
        # Perform annealing, get a new data point via SA, and augment a dataset.
        _, _, xast, yast = simulated_anneal(trained_FM_set, W, n_N, n_K) # SA
        # Augment the dataset with a new data point.
        xs = np.vstack((xs, xast)) # augmented x
        ys = np.append(ys, yast) # augmented y
    return trained_FM_set, xs, ys
 
def train_sub_nonstandard_SA(x: np.ndarray, y: np.ndarray, W: np.ndarray, n_N: int, n_K: int, Stand_Deviate: float, seed_Torch: int, num_it: int, R: float, num_seed: int
                            ) -> Tuple[Tuple[float, Tuple[np.ndarray, np.ndarray, float]], np.ndarray, np.ndarray]:
    """
    Run non-standardized SFMA via SA.

    Parameters:
        x (np.ndarray): Initial inputs of a dataset.
        y (np.ndarray): Initial outputs of a dataset.
        W (np.ndarray): W matrix.
        n_N (int): Number of rows of M.
        n_K (int): Number of columns of M.
        Stand_Deviate (float): Standard deviation.
        seed_Torch (int): Random seed.
        num_it (int): Number of the iterations of SFMA.
        R (float): Ratio.
        num_seed (int): Random seed.
 
    Returns: 
        Tuple[Tuple[float, Tuple[np.ndarray, np.ndarray, float]], np.ndarray, np.ndarray]:
        A (float): Final loss function. 
        B (np.ndarray): Final FM model parameter v.
        C (np.ndarray): Final FM model parameter w.
        D (float): Final FM model parameter w0.
        E (np.ndarray): Final input data x.
        F (np.ndarray): Final output data y.
    """

    Nbit = n_N*n_K # number of spins (qubits)
    kFM = int(Nbit/2)-1 # hyperparameter for a FM model
    xs, ys = x, y # dataset to be updated
    x_sub, y_sub = subdata_uniform(xs, ys, R, num_seed) # initial subdataset
    for i in trange(num_it): # SFMA loop
        # Train the FM model.
        torch.manual_seed(seed_Torch) # set a random seed for reproducibility 
        model = TorchFM(d=Nbit, k=kFM, Stand_Dev=Stand_Deviate) # instantiate a FM model
        trained_FM_set = train_FM_model(x_sub, y_sub, model) # train a FM model
        # Perform annealing, get a new data point via SA, and augment a dataset.
        _, _, xast, yast = simulated_anneal(trained_FM_set, W, n_N, n_K) # SA
        # Augment the dataset with a new data point and create a new subdataset.
        xs = np.vstack((xs, xast)) # augmented x
        ys = np.append(ys, yast) # augmented y
        x_sub, y_sub = subdata_uniform(xs, ys, R, num_seed) # new subdataset
    return trained_FM_set, xs, ys

### Function for Quantum Annealing (QA)

# Create a D-Wave sampler. 
my_token = 'my_token' # Set your D-Wave API token here
system_choice = 'Advantage_system6.4'  # D-Wave Systems' quantum device 
sampler = DWaveSampler(token=my_token, solver=system_choice) # sampler
embedded_sampler = EmbeddingComposite(sampler) # minor-embedding function provided in the Ocean SDK

# Quantum annealer 
def quantum_anneal(trained_FMmodel_set: Tuple[float, Tuple[np.ndarray, np.ndarray, float]], W: np.ndarray, n_N: int, n_K: int
                  ) -> Tuple[np.ndarray, float]:

    """
    QA for acquiring the best candidate (optimal) solution (NumPy array) and the corresponding objective value (float).

    Paramters:
        trained_FMmodel_set (Tuple[float, Tuple[np.ndarray, np.ndarray, float]]): (loss function, FM model parameters (v, w, w0)).
        W (np.ndarray): W matrix.
        n_N (int): Number of rows of M.
        n_K (int): Number of columns of M.

    Returns: 
        Tuple[np.ndarray, float]:
        np.ndarray: xhat (Best candidate (optimal) solution).
        float: yhat (Objective value of the best candidate (optimal) solution).
    """

    # Acquire parameters v, w, w0 from TorchFM and construct a QUBO matrix.
    _, trained_parameters = trained_FMmodel_set # (v,w,w0): drop a loss function
    v, w, w0 = trained_parameters # v, w, and w0
    Q_qubo = construct_QUBO(w, v) # QUBO matrix
    
    # Program QA
    res_qa = embedded_sampler.sample_qubo(Q_qubo, num_reads=nreads_QA) # run QA
    nbit = n_N*n_K # number of spins (qubits)
    xhat = sample_to_vector(res_qa.first.sample, nbit) # best candidate (optimal) solution
    
    # check if a valid solution was returned
    if len(xhat) == 0:
        raise RuntimeError("No solution was found.")

    yhat = blackbox(W, xhat, n_N, n_K) # corresponding objective value
    return xhat, yhat

# Training function with a subdataset 
def train_sub_standard_QA(x: np.ndarray, y: np.ndarray, W: np.ndarray, n_N: int, n_K: int, Mean: float, Stand_Deviate: float, seed_Torch: int, num_it: int, R: float, num_seed: int
                         ) -> Tuple[Tuple[float, Tuple[np.ndarray, np.ndarray, float]], np.ndarray, np.ndarray]:
    """
    Run standardized SFMA via QA.

    Parameters:
        x (np.ndarray): Initial inputs of a dataset.
        y (np.ndarray): Initial outputs of a dataset.
        W (np.ndarray): W matrix.
        n_N (int): Number of rows of M.
        n_K (int): Number of columns of M.
        Mean (float): Mean.
        Stand_Deviate (float): Standard deviation.
        seed_Torch (int): Random seed.
        num_it (int): Number of the iterations of SFMA.
        R (float): Ratio.
        num_seed (int): Random seed.
 
    Returns: 
        Tuple[Tuple[float, Tuple[np.ndarray, np.ndarray, float]], np.ndarray, np.ndarray]:
        A (float): Final value of a loss function. 
        B (np.ndarray): Final FM model parameter v.
        C (np.ndarray): Final FM model parameter w.
        D (float): Final FM model parameter w0.
        E (np.ndarray): Final input data x.
        F (np.ndarray): Final output data y.      
    """

    Nbit = n_N*n_K # number of spins (qubits)
    kFM = int(Nbit/2)-1 # hyperparameter for a FM model 
    xs, ys = x, y # dataset to be updated
    x_sub, y_sub = subdata_uniform(xs, ys, R, num_seed) # initial subdataset
    for i in trange(num_it): # SFMA loop
         # Train the FM model.       
        y_sub_stand = standard(y_sub, Mean, Stand_Deviate) # standardization
        torch.manual_seed(seed_Torch) # set a random seed for reproducibility 
        model = TorchFM(d=Nbit, k=kFM, Stand_Dev=Stand_Deviate) # instantiate a FM model
        trained_FM_set = train_FM_model(x_sub, y_sub_stand, model) # train a FM model
        # Perform annealing, get a new data point via QA, and augment data.
        xast, yast = quantum_anneal(trained_FM_set, W, n_N, n_K) # QA
        # Augment the dataset with a new data point and create a new subdataset.
        xs = np.vstack((xs, xast)) # augmented x
        ys = np.append(ys, yast) # augmented y
        x_sub, y_sub = subdata_uniform(xs, ys, R, num_seed) # new subdataset 
    return trained_FM_set, xs, ys

### Function for running random sampling (RS).

def Aug_RS_with_xin(W: np.ndarray, n_N: int, n_K: int, x_in: np.ndarray, n_aug: int, n_seed: int) -> Tuple[np.ndarray, np.ndarray]:   
                                    
    """
    Calculate an augmented dataset (a tuple (x_aug, y_aug)) via RS such that the first n_0 (int) elements of input data are equal to an original inital input data x_in (NumPy array).
    The rest of the elements are randomly sampled without duplicates using the same random seed used for generating the original inital dataset: n_seed (int).
    The size of an augmented (final) dataset (x_aug, y_aug) is equal to n_aug (int). 

    Parameters:
        W (np.ndarray): W matrix.
        n_N (int): Number of rows of M.
        n_K (int): Number of columns of M.
        x_in (np.ndarray): Original initial input whose data size is equal to the number of spins (qubits)).
        n_aug (int): Size of the final dataset.
        n_seed: Random seed.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Final dataset (x_aug, y_aug).
        x_aug (np.ndarray): Final input data.
        y_aug (np.ndarray): Final output data.
    """
    
    dim = x_in.shape[1] # number of spins (qubits)
    n_0 = x_in.shape[0] # number of sampes of an inital dataset 

    # Check if augmentation was done properly.
    if n_aug <= n_0:
        raise ValueError("n_aug must be greater than n_0.")

    # Return all dim-dimensional binary vectors and their indices.
    all_vectors = all_binaries(dim) # all binary vectors 
    x_in_indices = np.array([binary_array_to_decimal(vec) for vec in x_in]) # convert x_in to decimal: indices of x_in
    
    # Indices for generating new random binariy vectors  
    all_indices = np.arange(2**dim) # all indices
    remaining_indices = np.setdiff1d(all_indices, x_in_indices) # the remaining 

    # Randomly sample n_aug - n_0 unique indices from the remaining indices for the augmentation. 
    rg = np.random.default_rng(n_seed) # random number generator with n_seed
    additional_indices = rg.choice(remaining_indices, size = n_aug - n_0, replace=False) # n_aug - n_0 indices 

    # Combine x_in with extra binary vectors to form a random augmented dataset.   
    extra_vectors = all_vectors[additional_indices] # extra vectors
    x_aug = np.vstack((x_in, extra_vectors)) # augmented x
    y_aug = np.array([blackbox(W, xi, n_N, n_K) for xi in x_aug])  # augmented y

    return x_aug, y_aug


### Functions for calculating statistical quantities.

def output_sample_min(ensmeble_y: np.ndarray) -> np.ndarray:
    """
    Calculate samples of a minimum output vector (NumPy array).

    Parameters:
        ensmeble_y (np.ndarray): Final output data.
        
    Returns:
        np.ndarray: Minimum output data.      
    """
    
    list_y_min = [] # list for the row vectors of the accumulated minimum outputs with respect to the datasize (number of iterations): samples of y_min
    for array in ensmeble_y:
        y_min = np.minimum.accumulate(array) # array of the accumulated mimimum outputs 
        list_y_min.append(y_min) # append y_min to list_y_min
    ensmeble_ys_min = np.array(list_y_min) # convert list_y_min into a NumPy array 
    return ensmeble_ys_min

def success_rate(ensmeble_y: np.ndarray, n_0: int, n_total: int, BB_1st_min: float, n_sample: int) -> np.ndarray:
    """
    Calculate an array of success rates (NumPy array) obtained by SFMA or FMA.

    Parameters:
        ensmeble_y (np.ndarray): Samples of output data vectors.
        n_0 (int): Size of an inital dataset. 
        n_total (int): Size of a final dataset.
        BB_1st_min (float): Objective value of the optimal solution.  
        n_sample (int): Number of samples of a final dataset.
        
    Returns:
        np.ndarray: Array of success rates.     
    """
    
    ensmeble_y_min = output_sample_min(ensmeble_y) # minimum output vector
    l_rate_min = [] # list for success rates
    for a in range(n_0,n_total):
        l_min = list(ensmeble_y_min[:,a]) # convert the ath column of ensmeble_y_min into a list
        l_rate_min.append(l_min.count(BB_1st_min)/n_sample) # count the number of elements which are equal to BB_1st_min (the 1st optimal output) and divide by n_sample
    rate_min = np.array(l_rate_min) # convert l_rate_min into a NumPy array
    return rate_min

def success_rate_rounded(ensmeble_y: np.ndarray, n_0: int, n_total: int, n_rounded: int, BB_1st_min: float, n_sample: int) -> np.ndarray:
    """
    Calculate an array of success rates obtained by SFMA (FMA) rounded to n_round (int) decimal places (NumPy array).

    Parameters:
        ensmeble_y (np.ndarray): Samples of output data vectors.
        n_0 (int): Size of an inital dataset. 
        n_total (int): Size of a final dataset.
        n_rounded (int): Number of decimal places. 
        BB_1st_min (float): Objective value of the optimal solution.  
        n_sample (int): Number of samples of a final dataset.
        
    Returns:
        np.ndarray: Array of rounded success rates.      
    """
    ensmeble_y_min = output_sample_min(ensmeble_y) # minimum output vector
    BB_1st_min_rounded = np.round(BB_1st_min, n_rounded) # opt_1st rounded to n_round decimal places
    l_rate_min_rounded = [] # list for rounded success rates
    for a in range(n_0,n_total):
        l_min_rounded = list(np.round(ensmeble_y_min[:,a],n_rounded)) # convert the ath columns of ensmeble_y_min into a list
        l_rate_min_rounded.append(l_min_rounded.count(BB_1st_min_rounded)/n_sample) # count the number of elements which are equal to BB_1st_min_rounded (the 1st optimal output) and divide it by n_sample
    rate_min_rounded = np.array(l_rate_min_rounded) # convert l_rate_min_rounded into a NumPy array
    return rate_min_rounded

def find_min_location(s_rate: np.ndarray, target: float) -> int:
    """
    Calculate N_conv (int): Minimum number of iterations of SFMA (FMA) such that an updated success rate becomes equal to or greater than target (float).   

    Parameters:
        s_rate (np.ndarray): Array of success rates.
        target (float): Target value of success rates.
        
    Returns:
        int: N_conv.      
    """
    
    valid_elements = s_rate[s_rate >= target] # find all elements greater than or equal to target
    
    if valid_elements.size == 0:
        return None  # return None if no valid elements exist
    
    min_value = valid_elements.min() # find the minimum among these valid elements
    
    # get the index of this minimum value in the original array
    return np.where(s_rate == min_value)[0][0]  # return the first occurrence

# Function for searching ranges where elements of averaged minimum output vector calculated by FMA are equal to or greater than that obtained by SFMA.  
def find_ranges(diff_ys_mean: np.ndarray, condition: Callable[[np.ndarray], np.ndarray]) -> List[Tuple[int, int]]:
    """
    Calculate ranges of integers (list of (N_1, N_2) such that N_1 < N_2) where diff_ys_mean > 0.
    Two integers N_1 (int) and N_2 (int) represent the indices of an averaged minimum output vector.
    The quantity diff_ys_mean (NumPy array) describes the difference between an averaged minimum output vector calculated by FMA and that obtained by SFMA.

    Parameters:
        diff_ys_mean (np.ndarray): Difference between an averaged minimum output vector calculated by FMA and that obtained by SFMA.
        condition (Callable[[np.ndarray], np.ndarray]): Lambda function that returns a boolean mask representing where a given condition holds.
        
    Returns:
        List[Tuple[int, int]]: Tuples of ranges (N_1, N_2) where diff_ys_mean > 0.      
    """

    condition = lambda diff_y: diff_y >= 0 # lambda function 
    mask = condition(diff_ys_mean) # boolean mask 
    positions = np.where(mask)[0] # array of the positions of "True" in mask, 

    # Calculate tuples of ranges.
    ranges = [] # list for tuples
    if positions.size > 0:
        # starting from the first location of positions
        start = positions[0] 
        prev = positions[0]
        
        # if clause: if "True" appears consecutively in positions (i == prev + 1), update prev (prev = i) while fixing start   
        # else clause: when the consecutive appearance of "True" ends, create (start, prev), append it to ranges, and reset start and prev  
        for i in positions[1:]:
            if i == prev + 1:
                prev = i
            else:
                ranges.append((start, prev))
                start = prev = i
        ranges.append((start, prev))

    return ranges 