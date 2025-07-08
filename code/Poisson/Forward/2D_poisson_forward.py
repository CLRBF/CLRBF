import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import scipy.io
import random
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist

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


def gaussian_kernel(x, y, sigma):
    pairwise_dists = cdist(x, y, 'euclidean')
    return np.exp(-(sigma**2 * pairwise_dists**2))

def gaussian_kernel_grad_1(x, y, sigma):
    pairwise_dists = cdist(x, y, 'euclidean')
    return 2*sigma**2*(-2 + 2*sigma**2 * pairwise_dists**2) * np.exp(-(sigma**2 * pairwise_dists**2))


def generate_beta(N,sigma,input_x):

    # Kansas data
    N = N
    sigma = sigma
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()]).T

    all_points = points
    # interior linear system
    mask = (np.abs(points[:, 0]) != 1) & (np.abs(points[:, 1]) != 1)
    x_interior = points[mask]
    matrix_interior = gaussian_kernel_grad_1(x_interior, all_points, sigma)

    # boundary linear system
    boundary_mask = (np.abs(points[:, 0]) == 1) | (np.abs(points[:, 1]) == 1)
    boundary_points = points[boundary_mask]
    matrix_boundary = gaussian_kernel(boundary_points, all_points, sigma)

    # combined linear system
    matrix_combined = np.vstack([matrix_interior, matrix_boundary])
    b = np.concatenate((input_x*np.ones(matrix_interior.shape[0]), np.zeros(matrix_boundary.shape[0])), axis=0)

    # solution
    beta = np.linalg.solve(matrix_combined, b)
    return beta


seed_everything(42)

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
    # Initialize the class
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

    def cal_nlpos(self, train_x,train_beta, test_x):

        train_x = train_x.reshape(-1,1).to(device)
        train_x.requires_grad_(True)

        train_beta = train_beta.to(device)
        train_beta.requires_grad_(True)

        test_x = test_x.reshape(-1,1).to(device)
        test_x.requires_grad_(True)

        """training"""
        predict_beta = self.neural_net_f(train_x)
        predict_beta_save = predict_beta
        predict_beta =  predict_beta.reshape(-1,1)
        train_beta = train_beta.reshape(-1,1)


        loss1 = torch.mean(torch.square(predict_beta-train_beta)).to(device)

        loss = loss1

        predict_beta_test = self.neural_net_f(test_x)

        return loss, predict_beta_test,predict_beta_save


def sgmcmc(pars, net,train_x,train_beta,test_x):
    optimizer = torch.optim.Adam(net.parameters(), lr=pars.lr)
    loss_hist = np.zeros((int(pars.sn / 10), 2))
    count = 0
    train_x = Variable(torch.Tensor(train_x))
    train_beta = Variable(torch.Tensor(train_beta))
    test_x = Variable(torch.Tensor(test_x))

    for epoch in range(1, pars.sn + 1):
        net.train()
        net.to(device)
        optimizer.zero_grad()
        loss, predict_beta_test,predict_beta  = net.cal_nlpos(train_x,train_beta,test_x)
        # train loss
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print('\nEpoch {}  Loss: {:0.3f}'.format(epoch, loss.cpu().detach().numpy()))  # [0]
            loss_hist[count, :][:, None] = np.row_stack((loss.cpu().detach().numpy(),loss.cpu().detach().numpy()))
            count = count + 1

    return loss_hist, loss, predict_beta_test,predict_beta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameters')
    parser.add_argument('-N', default=20, type=int, help='total training Epochs')
    parser.add_argument('-sn', default=20000, type=int, help='total training Epochs')
    parser.add_argument('-lr', default=0.001, type=float, help='lr')
    parser.add_argument('-neuron', default=128, type=int, help='neuron')
    parser.add_argument('-in_dim', default=1, type=int, help='in_dim')
    parser.add_argument('-hidden_dim', default=128, type=int, help='hidden_dim')
    parser.add_argument('-sigma', default=15, type=int, help='parameters')
    parser.add_argument('-out_dim', default=2500, type=int, help='out_dim')
    parser.add_argument('-layernumber', default=10, type=int, help='layernumber')

    pars = parser.parse_args()
    N = pars.N
    sn = pars.sn
    lr = pars.lr
    neuron = pars.neuron
    layernumber = pars.layernumber
    in_dim = pars.in_dim
    hidden_dim = pars.hidden_dim
    out_dim = N*N
    num_train = 100
    sigma = pars.sigma

    #training set and its label
    train_x = np.linspace(-1, 1, num_train)
    training_beta = []
    for i in range(num_train):
        print(i)
        training_beta.append(generate_beta(N, sigma, train_x[i]))
    training_beta = np.array(training_beta)

    net = PhysicsInformedNN(neuron,layernumber,in_dim, hidden_dim, out_dim)
    net = net.apply(init_weights)
    loss_hist, loss,predict_beta_test,predict_beta = sgmcmc(pars, net,train_x,training_beta,test_x)









