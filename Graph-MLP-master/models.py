import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Linear, LayerNorm

class Mlp(nn.Module):
    def __init__(self, input_dim, hid_dim, dropout):
        super(Mlp, self).__init__()
        self.fc1 = Linear(input_dim, hid_dim)
        self.fc2 = Linear(hid_dim, hid_dim)
        self.act_fn = torch.nn.functional.gelu
        self._init_weights()

        self.dropout = Dropout(dropout)
        self.layernorm = LayerNorm(hid_dim, eps=1e-6)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.layernorm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x@x.T
    mask = torch.eye(x_dis.shape[0]).cuda()
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis*(x_sum**(-1))
    x_dis = (1-mask) * x_dis
    return x_dis

def get_contrastive_dis(x,y):
    """
    x :           batch_size x nhid
    y :           batch_size x nhid
    xy_dis(i,j):  item means the similarity between x(i) and y(j).
    """
    xy_dis = x@y.T
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    y_sum = torch.sum(y ** 2, 1).reshape(-1, 1)
    y_sum = torch.sqrt(y_sum).reshape(-1, 1)
    xy_sum = x_sum @ y_sum.T
    xy_dis = xy_dis*(xy_sum**(-1))
    return xy_dis

class GMLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GMLP, self).__init__()
        self.nhid = nhid
        self.mlp = Mlp(nfeat, self.nhid, dropout)
        self.classifier = Linear(self.nhid, nclass)

    def forward(self, x):
        x = self.mlp(x)

        feature_cls = x
        Z = x

        if self.training:
            x_dis = get_feature_dis(Z)

        class_feature = self.classifier(feature_cls)
        class_logits = F.log_softmax(class_feature, dim=1)

        if self.training:
            return class_logits, x_dis
        else:
            return class_logits

class MLPsimple(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MLPsimple, self).__init__()
        self.nhid = nhid
        self.mlp = Mlp(nfeat, self.nhid, dropout)
        self.classifier = Linear(self.nhid, nclass)

    def forward(self, x):
        x = self.mlp(x)
        feature_cls = x
        class_feature = self.classifier(feature_cls)
        class_logits = F.log_softmax(class_feature, dim=1)
        return class_logits
        
class VAE(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlatent, dropout):
        super(VAE, self).__init__()
        self.nhid = nhid
        self.nlatent = nlatent
        self.fc1 = Linear(nfeat, 200)
        self.fc2 = Linear(200, self.nhid)
        self.fc21 = Linear(self.nhid, self.nlatent)
        self.fc22 = Linear(self.nhid, self.nlatent)
        self.fc3 = Linear(self.nlatent, 200)
        self.fc4 = Linear(200, nfeat)
        self.act_fn = torch.nn.functional.gelu
        self._init_weights()
        
        self.dropout = Dropout(dropout)
        self.layernorm = LayerNorm(200, eps=1e-6)
        # self.classifier1 = Linear(self.nlatent, 200)
        # self.classifier2 = Linear(200, nclass)
        self.classifier3 = Linear(self.nlatent, nclass)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc21.weight)
        nn.init.xavier_uniform_(self.fc22.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc21.bias, std=1e-6)
        nn.init.normal_(self.fc22.bias, std=1e-6)
        nn.init.normal_(self.fc3.bias, std=1e-6)
        nn.init.normal_(self.fc4.bias, std=1e-6)
        
    def encode(self,x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.layernorm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.act_fn(x)
        return self.fc21(x),self.fc22(x)
        
    def decode(self,x):
        x = self.fc3(x)
        x = self.act_fn(x)
        x = self.fc4(x)
        x = F.sigmoid(x)
        return x

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to('cuda:0')
        return eps.mul(std).add_(mu)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)

        if self.training:
            x_dis = get_feature_dis(z)

        s = self.classifier3(z)
        class_logits = F.log_softmax(s, dim=1)
        s2 = self.classifier3(mu)
        class_logits2 = F.log_softmax(s2, dim=1)
        if self.training:
            return self.decode(z), mu, logvar, class_logits, x_dis, z
        else:
            return class_logits
            # return class_logits2
     


