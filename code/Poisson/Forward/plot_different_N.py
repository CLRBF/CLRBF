

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


N_values = [400, 625,900,1225, 1600,2025, 2500]
x_positions = np.linspace(0, 1, len(N_values))  # 0到1之间均匀分布的6个值

relative_L2_error_true_and_NN_array = [0.9812,0.8936, 0.410,0.0403,0.007938,0.0152, 0.0196]

plt.plot(x_positions, relative_L2_error_true_and_NN_array,marker='o', linestyle='-',markersize=10)


for xi, yi in zip(x_positions, relative_L2_error_true_and_NN_array):
    plt.text(xi, yi, f'{yi:.3f}', fontsize=12, ha='center', va='bottom',rotation=30)  


plt.xlabel('N',fontsize=15)
plt.ylabel('error',fontsize=15)
plt.xticks(x_positions, N_values)

plt.grid(linestyle='--', linewidth=0.5, alpha=0.3)

plt.tight_layout()


plt.show()