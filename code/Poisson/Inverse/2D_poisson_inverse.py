import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import scipy.io
import random
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist

np.set_printoptions(precision=50)
torch.set_printoptions(precision=50)
scaler = MinMaxScaler()
criterion = nn.MSELoss()

N = 50  

x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, y)
points = np.vstack([X.ravel(), Y.ravel()]).T

all_points = points

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

    def cal_nlpos(self, train_x,train_beta, test_x,x_guess,true_solution,matrix_solution ):

        x_guess = x_guess.to(device)

        train_x = train_x.reshape(-1,1).to(device)
        train_x.requires_grad_(True)

        train_beta = train_beta.to(device)
        train_beta.requires_grad_(True)

        x_guess = x_guess.reshape(-1, 1).to(device)
        x_guess.requires_grad_(True)

        """training"""
        predict_beta = self.neural_net_f(train_x)
        predict_beta_save = predict_beta
        predict_beta =  predict_beta.reshape(-1,1)
        train_beta = train_beta.reshape(-1,1)

        loss1 = torch.mean(torch.square(predict_beta-train_beta)).to(device)

        predict_beta_test = self.neural_net_f(x_guess).reshape(-1,1).float().to(device)
        matrix_solution = torch.tensor(matrix_solution).float().to(device)
        reconstructed_u = matrix_solution @ predict_beta_test
        true_solution = torch.tensor(true_solution).float().reshape(-1,1).to(device)
        loss2 = torch.mean(torch.square(reconstructed_u - true_solution)).to(device)


        loss = loss1 + loss2



        return loss, predict_beta_test,predict_beta_save, x_guess


def sgmcmc(pars, net,train_x,train_beta,test_x,x_guess,true_solution,matrix_solution):

    #todo reivised
    optimizer = torch.optim.Adam([{'params': net.parameters()}, {'params': [x_guess]}], lr=pars.lr)
    loss_hist = np.zeros((int(pars.sn / 10), 3))
    count = 0
    train_x = Variable(torch.Tensor(train_x))
    train_beta = Variable(torch.Tensor(train_beta))
    test_x  =  Variable(torch.Tensor(np.array([test_x])))

    x_guess = x_guess
    true_solution = true_solution
    matrix_solution = matrix_solution

    for epoch in range(1, pars.sn + 1):
        net.train()
        net.to(device)
        optimizer.zero_grad()
        loss, predict_beta_test,predict_beta, x_guess  = net.cal_nlpos(train_x,train_beta,test_x,x_guess,true_solution,matrix_solution )
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print('\nEpoch {}  Loss: {:0.3f}'.format(epoch, loss.cpu().detach().numpy()))  # [0]
            loss_hist[count, :][:, None] = np.row_stack((loss.cpu().detach().numpy(),loss.cpu().detach().numpy(),x_guess.cpu().detach().numpy()))
            count = count + 1

            print('x_guess',x_guess)

    return loss_hist, loss, predict_beta_test,predict_beta


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Hyperparameters')
    parser.add_argument('-sn', default=50000, type=int, help='total training Epochs')
    parser.add_argument('-lr', default=0.001, type=float, help='sampling learning rate (default for pruning)')
    parser.add_argument('-neuron', default=128, type=int, help='sampling learning rate (default for pruning)')
    parser.add_argument('-in_dim', default=1, type=int, help='sampling learning rate (default for pruning)')
    parser.add_argument('-hidden_dim', default=128, type=int, help='sampling learning rate (default for pruning)')
    parser.add_argument('-out_dim', default=N**2, type=int, help='sampling learning rate (default for pruning)')
    parser.add_argument('-layernumber', default=10, type=int, help='sampling learning rate (default for pruning)')

    pars = parser.parse_args()
    sn = pars.sn
    lr = pars.lr
    neuron = pars.neuron
    layernumber = pars.layernumber
    in_dim = pars.in_dim
    hidden_dim = pars.hidden_dim
    out_dim = pars.out_dim
    num_train = 100
    sigma = 15

    train_x = np.linspace(-1, 1, num_train)
    training_beta = []
    for i in range(num_train):
        training_beta.append(generate_beta(N, sigma, train_x[i]))
    training_beta = np.array(training_beta)

    test_x = np.random.uniform(-0.9,1,1)
    # test_x = np.array([-0.18837377419001133])
    print('test_x',test_x)

    true_beta_test_x = generate_beta(N, sigma, test_x)
    matrix_solution = gaussian_kernel(all_points, all_points, sigma)
    true_solution = matrix_solution @ true_beta_test_x

    x_guess =  torch.tensor([1.0], requires_grad=True)

    net = PhysicsInformedNN(neuron,layernumber,in_dim, hidden_dim, out_dim)
    net = net.apply(init_weights)
    print(net)
    loss_hist, loss,predict_beta_test,predict_beta = sgmcmc(pars, net,train_x,training_beta,test_x,x_guess,true_solution,matrix_solution)




