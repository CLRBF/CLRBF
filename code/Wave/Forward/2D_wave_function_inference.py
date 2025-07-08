

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from matplotlib.ticker import ScalarFormatter
import scipy.io
from matplotlib.gridspec import GridSpec
import argparse

import numpy as np
import torch
import torch.nn as nn
import sys
from torch.autograd import Variable
import argparse
import scipy.io
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
import time

np.set_printoptions(precision=20)
torch.set_printoptions(precision=20)
scaler = MinMaxScaler()
criterion = nn.MSELoss()

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("cuda is available")
else:
    device = torch.device('cpu')
    print("cpu is available")



def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(42)

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def gaussian_kernel(x, y, sigma):
    pairwise_dists = torch.tensor(cdist(x, y, 'euclidean')).to(device)
    # return np.exp(-(sigma**2 * pairwise_dists**2))
    return torch.exp(-(sigma**2 * pairwise_dists**2)).to(device)

def gaussian_kernel_t(x, y, sigma):
    pairwise_dists = torch.tensor(cdist(x, y, 'euclidean'))
    x = torch.tensor(x).to(device)
    y = torch.tensor(y).to(device)
    result = -2*sigma**2*(x[:,0].unsqueeze(-1)-y[:,0].unsqueeze(0))*np.exp(-(sigma**2 * pairwise_dists**2))
    return result.to(device)

def gaussian_kernel_grad_tt_s1s1_s2s2(x, y, sigma):
    x = torch.tensor(x).to(device)
    y = torch.tensor(y).to(device)

    pairwise_dists = torch.tensor(cdist(x, y, 'euclidean')).to(device)
    
    result = (2*sigma**2
            + (2*sigma**2)**2
            * ((x[:,0].unsqueeze(-1)-y[:,0].unsqueeze(0))**2
            -((x[:,1].unsqueeze(-1)-y[:,1].unsqueeze(0))**2
                +(x[:,2].unsqueeze(-1)-y[:,2].unsqueeze(0))**2)))\
        * np.exp(-(sigma**2 * pairwise_dists**2))
    return result.to(device)


def get_u_kansa(N,Nt,sigma,T_final,x_postion):
    x_position = x_postion
    t = np.linspace(0, T_final, Nt)
    s1 = np.linspace(0, 1, N)
    s2 = np.linspace(0, 1, N)
    h = s1[1] - s2[0]

    T, S1, S2 = np.meshgrid(t, s1, s2, indexing='ij')
  
    T_flatten = T.flatten()
    S1_flatten = S1.flatten()
    S2_flatten = S2.flatten()


    collocation_points = np.column_stack((T_flatten, S1_flatten, S2_flatten))

    """kansa's method by gaussian kernel"""
    points = collocation_points

    'points choosen'
    mask_initial = (np.abs(points[:,0]) == 0) & (np.abs(points[:,1]) != 1) & (np.abs(points[:, 1]) != 0) & (np.abs(points[:,2]) != 0) & (np.abs(points[:,2]) != 1)
    x_initial = points[mask_initial]
    mask_interior = (np.abs(points[:,1]) != 1) & (np.abs(points[:, 1]) != 0) & (np.abs(points[:,2]) != 0) & (np.abs(points[:,2]) != 1) & (np.abs(points[:,0]) != 0)
    x_interior = points[mask_interior]
    mask_boundary = (np.abs(points[:,1]) == 1) | (np.abs(points[:, 1]) == 0) | (np.abs(points[:,2]) == 0) | (np.abs(points[:,2]) == 1)
    x_boundary = points[mask_boundary]


    matrix_boundary = gaussian_kernel(x_boundary, points, sigma)
    x_initial_1 = x_initial[1::2]
    x_initial_2 = x_initial[::2]

    matrix_initial_1 = gaussian_kernel(x_initial_1, points, sigma)
    matrix_initial_2 = gaussian_kernel_t(x_initial_2, points, sigma)

    matrix_interior =  gaussian_kernel_grad_tt_s1s1_s2s2(x_interior, points, sigma)
    matrix_combined = torch.cat([matrix_interior, matrix_boundary, matrix_initial_1, matrix_initial_2], dim=0).to(device)
    # matrix_combined = np.vstack([matrix_interior, matrix_boundary, matrix_initial_1, matrix_initial_2])
    
    #right hand side
    x_interior = torch.tensor(x_interior).to(device)
    x_boundary = torch.tensor(x_boundary).to(device)
    x_initial_1 = torch.tensor(x_initial_1).to(device)

    b_interior = x_position*(2*x_interior[:,1]*(1-x_interior[:,1]) + 2*x_interior[:,2]*(1-x_interior[:,2]) - c**2*x_interior[:,1]*(1-x_interior[:,1])*x_interior[:,2]*(1-x_interior[:,2]))*np.cos(c*x_interior[:,0])
    b_interior = b_interior.to(device)

    b_initial_1 = x_position*x_initial_1[:,1]*(1-x_initial_1[:,1])*x_initial_1[:,2]*(1-x_initial_1[:,2])
    b_initial_1 = b_initial_1.to(device)
    b_initial_2 = torch.zeros(matrix_initial_2.shape[0]).to(device)
   
  
    b_bc = x_position*x_boundary[:,1]*(1-x_boundary[:,1])*x_boundary[:,2]*(1-x_boundary[:,2])*np.cos(c*x_boundary[:,0])
    b_bc = b_bc.to(device)


    b = torch.cat([b_interior, b_bc, b_initial_1, b_initial_2], dim=0).to(device)
    # b = np.concatenate((b_interior, b_bc, b_initial_1, b_initial_2), axis=0)

    start = time.time()
    beta = torch.linalg.solve(matrix_combined, b)
    end = time.time()
    print('time',end-start)

    print('x_postion',x_postion)
    beta_minus_mean = beta - 0.1*beta.mean()
    print('beta_new',beta_minus_mean)
    matrix_solution = gaussian_kernel(points, points, sigma)

    u_approx = matrix_solution @ beta
    u_approx = u_approx.reshape(Nt, N, N)
    u_approx_beta_minus_mean = matrix_solution @ beta_minus_mean
    u_approx_beta_minus_mean = u_approx_beta_minus_mean.reshape(Nt, N, N)
    u_true = x_position*points[:,1]*(1-points[:,1])*points[:,2]*(1-points[:,2])*np.cos(c*points[:,0])
    u_true = torch.tensor(u_true).to(device)
    u_true_reshape = u_true.reshape(Nt, N, N)

    #error
    error = u_true_reshape - u_approx

    return u_approx, u_true_reshape, u_approx_beta_minus_mean, error, beta, points, N, Nt, sigma
        


class PhysicsInformedNN(torch.nn.Module):
    def __init__(self, neuron,layernumber,in_dim, hidden_dim, out_dim):
        super(PhysicsInformedNN, self).__init__()
        layers = []
        for i in range(layernumber - 1):
            if i == 0:
                layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
                layers.append(nn.Tanh())
            else:
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
                layers.append(nn.Tanh())
        layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))
        self.linear = nn.Sequential(*layers)

    def neural_net_f(self, x):
        return self.linear(x)

    def cal_nlpos(self, train_x,train_beta, test_x, beta_test_true):


        train_x = train_x.unsqueeze(-1).to(device).requires_grad_(True)
        train_beta = train_beta.to(device).squeeze(-1).requires_grad_(True)
        test_x = test_x.unsqueeze(-1).to(device).requires_grad_(True)

        """training"""
        predict_beta = self.neural_net_f(train_x)
        predict_beta_save = predict_beta.to(device)
        predict_beta =  predict_beta.reshape(-1,1)
        train_beta = train_beta.reshape(-1,1)
        loss1 = torch.mean(torch.square(predict_beta-train_beta)).to(device)
        loss = loss1

        """test"""
        start = time.time()
        predict_beta_test = self.neural_net_f(test_x).reshape(out_dim)
        end = time.time()
        print('clrbf_time',end-start)
        beta_test_true = beta_test_true.to(device).reshape(out_dim)
        loss_2 =  torch.mean(torch.square(predict_beta_test-beta_test_true)).to(device)

        return loss, predict_beta_test,predict_beta_save,loss_2

def sgmcmc(pars, net,train_x,train_beta,test_x, beta_test_true):
    optimizer = torch.optim.Adam(net.parameters(), lr=pars.lr)
    loss_hist = np.zeros((int(pars.sn / 10), 2))
    count = 0
    train_x = Variable(torch.Tensor(train_x))
    train_beta = Variable(torch.Tensor(train_beta))
    test_x = Variable(torch.Tensor(test_x))
    beta_test_true = beta_test_true
    # beta_test_true = Variable(torch.Tensor(beta_test_true))

    for epoch in range(1, pars.sn + 1):
        net.train()
        net.to(device)
        optimizer.zero_grad()
        loss, predict_beta_test,predict_beta,loss_2  = net.cal_nlpos(train_x,train_beta,test_x,beta_test_true)
        # train loss
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print('\nEpoch {}  Loss: {:0.3f}'.format(epoch, loss.cpu().detach().numpy()))  # [0]
            loss_hist[count, :][:, None] = np.row_stack((loss.cpu().detach().numpy(),loss_2.cpu().detach().numpy()))
            count = count + 1
            print('loss_2_MSE', loss_2)

    return loss_hist, loss, predict_beta_test,predict_beta,loss_2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameters')
    
    parser.add_argument('--N', default=10, type=int, help='number of collocation points')
    parser.add_argument('--Nt', default=5, type=int, help='number of temporal points')
    parser.add_argument('--sigma', default=21, type=int, help='kernel parameter')
    parser.add_argument('--c', default=10, type=int, help='c parameter')
    parser.add_argument('--x_position', default=1, type=float, help='c parameter')
    parser.add_argument('--T_final', default=0.05, type=float, help='c parameter')

    parser.add_argument('-sn', default=20000, type=int, help='total training Epochs')
    parser.add_argument('-lr', default=1e-3, type=float, help='sampling learning rate (default for pruning)')
    parser.add_argument('-neuron', default=128, type=int, help='sampling learning rate (default for pruning)')
    parser.add_argument('-in_dim', default=1, type=int, help='sampling learning rate (default for pruning)')
    parser.add_argument('-hidden_dim', default=128, type=int, help='sampling learning rate (default for pruning)')
    parser.add_argument('-out_dim', default=200, type=int, help='sampling learning rate (default for pruning)')
    parser.add_argument('-layernumber', default=5, type=int, help='sampling learning rate (default for pruning)')
    # parser.add_argument('-N', default=20, type=int, help='number of collocation points')
    parser.add_argument('-Ntrain', default=5, type=int, help='number of training (x_1_i,x_2_i) points')
    # parser.add_argument('-sigma', default=500, type=int, help='kernel parameter')



    pars = parser.parse_args()

    N = pars.N

    Nt = pars.Nt
    sigma = pars.sigma
    c = pars.c
    x_position = pars.x_position
    T_final = pars.T_final

    sn = pars.sn
    lr = pars.lr
    neuron = pars.neuron
    layernumber = pars.layernumber
    in_dim = pars.in_dim
    hidden_dim = pars.hidden_dim
    out_dim = pars.N * pars.N * pars.Nt #same as the number of the collcation points
    num_train = pars.Ntrain


        test_x = torch.tensor([0.747104810]).to(device)
    u_approx_test, u_true_reshape_test, u_approx_beta_minus_mean_test, error_test, beta_test, points_test, N_test, Nt_test, sigma_test = get_u_kansa(N,Nt,sigma,T_final,test_x)
    test_beta_true = beta_test

    #training data
    train_x = torch.linspace(0.5, 1, num_train).to(device)

    training_beta = torch.zeros((num_train,out_dim)).to(device)
    for i in range(num_train):
        x_position = train_x[i]
        u_approx, u_true_reshape, u_approx_beta_minus_mean, error, beta, points, N, Nt, sigma = get_u_kansa(N,Nt,sigma,T_final,x_position)
        training_beta[i] = beta
    training_beta = training_beta.to(device)
    print('beta_training_mean',training_beta.mean())


    """network"""
    net = PhysicsInformedNN(neuron,layernumber,in_dim, hidden_dim, out_dim)
    net = net.apply(init_weights)
    print('network structure', net)


    loss_hist, loss,predict_beta_test,predict_beta,loss_2 = sgmcmc(pars, net,train_x,training_beta,test_x, test_beta_true)



    












