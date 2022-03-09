import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init


class SparseLinear(nn.Module):

    def __init__(self, custom_mat, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        #self.custom_mat = torch.tensor(custom_mat).type(torch.FloatTensor)
        self.custom_mat = Parameter(torch.Tensor(custom_mat.T))

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def linear(self, input, weight, custom_mat, bias=None):
        """
        Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
        Shape:
            - Input: :math:`(N, *, in\_features)` where `*` means any number of
              additional dimensions
            - Weight: :math:`(out\_features, in\_features)`
            - Bias: :math:`(out\_features)`
            - Output: :math:`(N, *, out\_features)`
        """
        if input.dim() == 2 and bias is not None:
            # fused op is marginally faster
            weight = torch.mul(weight, custom_mat)
            return torch.addmm(bias, input, weight.t())

        weight = torch.mul(weight, custom_mat)
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        return output

    def forward(self, input):
        return self.linear(input, self.weight, self.custom_mat, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
