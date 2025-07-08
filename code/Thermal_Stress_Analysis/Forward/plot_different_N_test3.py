


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
from matplotlib.gridspec import GridSpec
import sys


N_values = [100, 2000, 10000, 15000, 21158]
x_positions = np.linspace(0, 1, len(N_values))  # 0到1之间均匀分布的6个值

relative_L2_error_true_and_NN_array = [0.039,0.018,0.015,0.013,0.013]

plt.plot(x_positions, relative_L2_error_true_and_NN_array,marker='o', linestyle='-',markersize=10)


for xi, yi in zip(x_positions, relative_L2_error_true_and_NN_array):
    plt.text(xi, yi, f'{yi:.3f}', fontsize=12, ha='center', va='bottom',rotation=30)  # 标签在点下方



plt.xlabel('N',fontsize=15)
plt.ylabel('error',fontsize=15)
plt.xticks(x_positions, N_values)

# plt.title('Scenario 1')
plt.grid(linewidth=0.5, alpha=0.3)

plt.tight_layout()


plt.show()