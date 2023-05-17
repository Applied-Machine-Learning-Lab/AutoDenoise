import torch

from torchfm.layer import FeaturesEmbedding, CrossNetwork, MultiLayerPerceptron, controller_mlp,kmax_pooling


class DeepCrossNetworkModel(torch.nn.Module):
    """
    A pytorch implementation of Deep & Cross Network.

    Reference:
        R Wang, et al. Deep & Cross Network for Ad Click Predictions, 2017.
    """

    def __init__(self, field_dims, embed_dim, num_layers, mlp_dims, dropout):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cn = CrossNetwork(self.embed_output_dim, num_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.linear = torch.nn.Linear(mlp_dims[-1] + self.embed_output_dim, 1)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)
        x_l1 = self.cn(embed_x)
        h_l2 = self.mlp(embed_x)
        x_stack = torch.cat([x_l1, h_l2], dim=1)
        p = self.linear(x_stack)
        return torch.sigmoid(p.squeeze(1))


class DeepCrossNetworkModel_Controller(torch.nn.Module):
    """
    A pytorch implementation of Deep & Cross Network.

    Reference:
        R Wang, et al. Deep & Cross Network for Ad Click Predictions, 2017.
    """

    def __init__(self, field_dims, embed_dim, num_layers, mlp_dims, dropout):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cn = CrossNetwork(self.embed_output_dim, num_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.linear = torch.nn.Linear(mlp_dims[-1] + self.embed_output_dim, 1)
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

        embed_x = embed_x.view(-1, self.embed_output_dim)
        x_l1 = self.cn(embed_x)
        h_l2 = self.mlp(embed_x)
        x_stack = torch.cat([x_l1, h_l2], dim=1)
        p = self.linear(x_stack)
        return torch.sigmoid(p.squeeze(1))



class DeepCrossNetworkModel_Controller_hard(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, num_layers, mlp_dims, dropout, k, device):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cn = CrossNetwork(self.embed_output_dim, num_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.linear = torch.nn.Linear(mlp_dims[-1] + self.embed_output_dim, 1)
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

        embed_x = embed_x.view(-1, self.embed_output_dim)
        x_l1 = self.cn(embed_x)
        h_l2 = self.mlp(embed_x)
        x_stack = torch.cat([x_l1, h_l2], dim=1)
        p = self.linear(x_stack)
        return torch.sigmoid(p.squeeze(1))