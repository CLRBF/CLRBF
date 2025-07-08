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

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(42)

def gaussian_kernel(x, y, sigma):
    pairwise_dists = torch.tensor(cdist(x, y, 'euclidean')).to(device)
    return torch.exp(-(sigma**2 * pairwise_dists**2)).to(device)

def generate_beta(sigma,blade_node_select,u_select):
    matrix_blade = gaussian_kernel(blade_node_select.T, blade_node_select.T, sigma)
    # solution
    u_select = torch.tensor(u_select).to(device)
    start_time = time.time()
    beta = torch.linalg.solve(matrix_blade, u_select).to(device)
    end_time = time.time()
    print(f"Time for Gaussian kernel computation: {end_time - start_time:.6f} seconds")
    return beta

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("cuda is available")
else:
    device = torch.device('cpu')
    print("cpu is available")

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

        train_x = train_x.to(device).requires_grad_(True)
        train_beta = train_beta.to(device).squeeze(-1).requires_grad_(True)
        test_x = test_x.unsqueeze(0).to(device).requires_grad_(True)

        """training"""
        predict_beta = self.neural_net_f(train_x)
        predict_beta_save = predict_beta.to(device)
        predict_beta =  predict_beta.reshape(-1,1)
        train_beta = train_beta.reshape(-1,1)
        loss1 = torch.mean(torch.square(predict_beta-train_beta)).to(device)
        loss = loss1

        """test"""
        start_time = time.time()
        predict_beta_test = self.neural_net_f(test_x).reshape(N)
        end_time = time.time()
        print(f"Time for NN computation: {end_time - start_time:.6f} seconds")
        beta_test_true = beta_test_true.to(device).reshape(N)
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
    parser.add_argument('-sn', default=20000, type=int, help='total training Epochs')
    parser.add_argument('-lr', default=1e-3, type=float, help='sampling learning rate (default for pruning)')
    parser.add_argument('-neuron', default=128, type=int, help='sampling learning rate (default for pruning)')
    parser.add_argument('-in_dim', default=2, type=int, help='sampling learning rate (default for pruning)')
    parser.add_argument('-hidden_dim', default=128, type=int, help='sampling learning rate (default for pruning)')
    parser.add_argument('-out_dim', default=200, type=int, help='sampling learning rate (default for pruning)')
    parser.add_argument('-layernumber', default=5, type=int, help='sampling learning rate (default for pruning)')
    parser.add_argument('-N', default=10000, type=int, help='number of collocation points')
    parser.add_argument('-Ntrain', default=5, type=int, help='number of training (x_1_i,x_2_i) points')
    parser.add_argument('-sigma', default=500, type=int, help='kernel parameter')

    pars = parser.parse_args()

    N = pars.N
    sn = pars.sn
    lr = pars.lr
    neuron = pars.neuron
    layernumber = pars.layernumber
    in_dim = pars.in_dim
    hidden_dim = pars.hidden_dim
    out_dim = pars.N #same as the number of the collcation points
    num_train = pars.Ntrain
    sigma = pars.sigma

    number = 1
    blade_data = scipy.io.loadmat(f'./blade_train/blade{number}.mat')
    blade_node = blade_data['nodes']

    index = np.int64(np.linspace(0,blade_node.shape[1]-1,N))



    '''new train_x, use the blade_test data'''
    train_x = scipy.io.loadmat(f'./blade_test/X_test.mat')
    train_x = train_x['X_test'][3:num_train,:]
    training_beta = torch.zeros((num_train-3,N,1)).to(device)

    for i in range(3,num_train):
        data = scipy.io.loadmat(f'./blade_test/blade_test{i+1}.mat')
        blade_node_select = blade_node[:,index]
        u_select = data['stress'][index,:]
        training_beta[i-3,] = generate_beta(sigma, blade_node_select, u_select)
    training_beta = training_beta.to(device)
    print('beta_training_mean',training_beta.mean())

    number_test = 1 
    test_x =  scipy.io.loadmat(f'./blade_test/X_test')
    test_x = test_x['X_test'][0,:]
    blade_data_test = scipy.io.loadmat(f'./blade_test/blade_test{number_test}.mat')
    print('test_x',test_x)
    u_select_test_data = blade_data_test['stress'][index,:]
    beta_test_true = generate_beta(sigma, blade_node_select, u_select_test_data) #kansa's result
    print('beta_test_true',beta_test_true)

    """network"""
    net = PhysicsInformedNN(neuron,layernumber,in_dim, hidden_dim, out_dim)
    net = net.apply(init_weights)
    print('network structure', net)
    loss_hist, loss,predict_beta_test,predict_beta,loss_2 = sgmcmc(pars, net,train_x,training_beta,test_x, beta_test_true)






