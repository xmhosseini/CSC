import torch
import torch.nn as nn
from torch import cat
import torch.nn.init as init
from numpy import sqrt, ceil
import numpy as np

from SparseLinear import SparseLinear

class SparseModel(nn.Module):
    def __init__(self, original_model, num_classes):
        super(SparseModel, self).__init__()

        self.features = original_model.features

        cm_11, cm_12 = getCustomMatrix(9216, 4096, 4096, 4096)
        cm_21, cm_22 = getCustomMatrix(4096, 4096, 4096, 1)
        cm_31, cm_32 = getCustomMatrix(4096, 1000, 1024, 1)

        self.classifier = nn.Sequential(
                        nn.Linear(9216, 4096),
                        # SparseLinear(cm_11, 9216, 4096),
						nn.BatchNorm1d(4096),
                        nn.ReLU(inplace=True),
                        nn.Linear(4096, 4096),
                        # SparseLinear(cm_12, 4096, 4096),
						nn.BatchNorm1d(4096),
                        nn.ReLU(inplace=True),
                        nn.Linear(4096, num_classes)
                    )

        #Freze conv layer weights
        for idx, m in enumerate(self.features.children()):
            for param in m.parameters():
                param.requires_grad = False

        self.classifier.apply(weights_init)

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), 256 * 6 * 6)
        y = self.classifier(f)
        return y

def getCustomMatrix(ip, op, n, c):

    rows1=ip
    N1=n;
    cols=op
    connectivity=c

    custom_matrix_1=[[0 for _ in range(N1)] for _ in range(N1)]
    diff_sets=[0, 1];
    # diff_sets=[0, 1, 2, 3, 4, 5, 6, 7];
    # diff_sets = [_ * 1 for _ in diff_sets]
    diff_sets=[];
    for i in range (0, int(sqrt(connectivity*N1))):
        diff_sets.append(i);
    for i in range (0, len(diff_sets)):
        custom_matrix_1[0][diff_sets[i]]=1;
    for i in range (1, N1):
        for j in range (0, N1):
            custom_matrix_1[i][j]=custom_matrix_1[i-1][(j-1)%(N1)]
    tile_times_1=int(ceil(rows1/N1))
    custom_matrix_1=np.tile(custom_matrix_1, (tile_times_1,1))
    custom_matrix_1=np.asarray(custom_matrix_1)
    custom_matrix_1=custom_matrix_1 [0:rows1,:]

    custom_matrix_L2=[[0 for _ in range(N1)] for _ in range(N1)]
    diff_sets = [_ * int(sqrt(connectivity*N1)/connectivity) for _ in diff_sets]
    # for i in range (0, N1):
    #     diff_sets.append(i);
    # diff_sets = [_ * int(sqrt(connectivity*N1)/connectivity) for _ in diff_sets]
    for i in range (0, len(diff_sets)):
        custom_matrix_L2[0][diff_sets[i]]=1;
    for i in range (1, N1):
        for j in range (0, N1):
            custom_matrix_L2[i][j]=custom_matrix_L2[i-1][(j-1)%(N1)]
    tile_times_2=int(ceil(cols/N1))
    custom_matrix_L2=np.tile(custom_matrix_L2, (1,tile_times_2))
    custom_matrix_L2=np.asarray(custom_matrix_L2)
    custom_matrix_L2=custom_matrix_L2 [:,0:cols]

    return custom_matrix_1, custom_matrix_L2

def weights_init(model):
    if type(model) in [SparseLinear, nn.Linear]:
        nn.init.xavier_normal_(model.weight.data)
        nn.init.constant_(model.bias.data, 0.1)
