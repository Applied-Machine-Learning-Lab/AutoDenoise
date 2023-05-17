import torch

from torchfm.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron, controller_mlp, kmax_pooling


class DeepFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of DeepFM.

    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))


class DeepFactorizationMachineModel_Controller(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        self.controller = controller_mlp(self.embed_output_dim, [len(field_dims)], dropout)
        self.BN = torch.nn.BatchNorm1d(len(field_dims))

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)

        '''controller'''
        embed_x = self.BN(embed_x)
        weight = self.controller(embed_x)
        embed_x = embed_x * torch.unsqueeze(weight,2)
        '''end controller'''

        x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))


class DeepFactorizationMachineModel_Controller_hard(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, k, device):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        self.controller = controller_mlp(self.embed_output_dim, [len(field_dims)], dropout)
        self.BN = torch.nn.BatchNorm1d(len(field_dims))
        self.k = k
        self.device = device

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)

        '''controller'''
        embed_x = self.BN(embed_x)
        weight = self.controller(embed_x)
        kmax_index, kmax_weight = kmax_pooling(weight,1,self.k)
        kmax_weight = kmax_weight/torch.sum(kmax_weight,dim=1).unsqueeze(1) #reweight, 使结果和为1
        mask = torch.zeros(weight.shape[0],weight.shape[1]).to(self.device)
        mask = mask.scatter_(1,kmax_index,kmax_weight) #填充对应索引位置为weight值
        embed_x = embed_x * torch.unsqueeze(mask,2)
        '''end controller'''

        x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))