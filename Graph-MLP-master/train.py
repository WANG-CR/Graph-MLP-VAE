from __future__ import division
from __future__ import print_function
import random
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import GMLP, VAE, MLPsimple, get_contrastive_dis
from utils import load_citation, accuracy, get_A_r
import warnings
warnings.filterwarnings('ignore')

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--epochs', type=int, default=400,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--data', type=str, default='cora',
                    help='dataset to be used')
parser.add_argument('--alpha', type=float, default=2.0,
                    help='To control the ratio of Ncontrast loss')
parser.add_argument('--batch_size', type=int, default=2048,
                    help='batch size')
parser.add_argument('--order', type=int, default=2,
                    help='to compute order-th power of adj')
parser.add_argument('--tau', type=float, default=1.0,
                    help='temperature for Ncontrast loss')
parser.add_argument('--result_dir_name', type=str, default="ckpts",
                    help='result directory of model dictionary')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

## get data
adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.data, 'AugNormAdj', True)
print(adj)
adj_label = get_A_r(adj, args.order)


## Model and optimizer
model = GMLP(
    nfeat=features.shape[1],
    nhid=args.hidden,
    nclass=labels.max().item() + 1,
    dropout=args.dropout,
)

optimizer = optim.Adam(
    model.parameters(), 
    lr=args.lr, 
    weight_decay=args.weight_decay
)
                       
##modelVAE and Optimizer                       
modelVAE = VAE(
    nfeat=features.shape[1],
    nhid=args.hidden,
    nclass=labels.max().item() + 1,
    nlatent=10,
    dropout=args.dropout,
)

optimizerVAE = optim.Adam(
    modelVAE.parameters(),
    lr=args.lr, 
    weight_decay=args.weight_decay
)

optimizerClassifier = optim.Adam(
    filter(lambda p: p.requires_grad, modelVAE.parameters()),
    lr=10*args.lr, 
    weight_decay=args.weight_decay
)

#a simple MLP
modelMLP = MLPsimple(
    nfeat=features.shape[1],
    nhid=args.hidden,
    nclass=labels.max().item() + 1,
    dropout=args.dropout,
)

optimizerMLP = optim.Adam(
    modelMLP.parameters(),
    lr=args.lr, 
    weight_decay=args.weight_decay
)

if args.cuda:
    model.cuda()
    modelVAE.cuda()
    modelMLP.cuda()
    features = features.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()




def Ncontrast(x_dis, adj_label, tau = 1):
    """
    compute the Ncontrast loss
    """
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log((x_dis_sum_pos+1e-8) * (x_dis_sum**(-1)+1e-8)).mean()
    #reasonable or not? smoothing both denominator and numerator
    return loss

def Ncontrast2(xy_dis, tau = 1):
    """
    compute the Ncontrast2 loss
    xy_dis = get_contrastive_dis(x,y)
    for batch of sample xi and yi (both sampled by VAE)
    for each i, consider sim(xi,yi)/sum(sim(xi,yj) as its loss
    same formula with that paper: https://github.com/princeton-nlp/SimCSE
    """
    xy_dis = torch.exp(tau * xy_dis)
    xy_dis_sum = torch.sum(xy_dis, 1)
    mask = torch.eye(xy_dis.shape[0]).cuda()
    xy_dis_sum_pos = torch.sum(xy_dis*mask, 1)
    loss = -torch.log((xy_dis_sum_pos+1e-8) * (xy_dis_sum**(-1)+1e-8)).mean()
    return loss



def vae_loss(recon_x, x, mu, logvar):
    vaeloss = nn.BCELoss()
    recon_loss = vaeloss(recon_x, x)
    kl_loss = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar, 1))
    return recon_loss+kl_loss 

def get_batch(batch_size):
    """
    get a batch of feature & adjacency matrix
    """
    rand_indx = torch.tensor(np.random.choice(np.arange(adj_label.shape[0]), batch_size)).cuda()
    rand_indx[0:len(idx_train)] = idx_train
    rand_indx = rand_indx.type(torch.LongTensor)
    ## idx_train: [0::139]
    # print(features)
    features_batch = features[rand_indx]
    adj_label_batch = adj_label[rand_indx,:][:,rand_indx]
    return features_batch, adj_label_batch

def train():
    features_batch, adj_label_batch = get_batch(batch_size=args.batch_size)
    model.train()
    optimizer.zero_grad()
    output1, x_dis1 = model(features_batch)
    output2, x_dis2 = model(features_batch)
    loss_train_class = F.nll_loss(output[idx_train], labels[idx_train])
    loss_Ncontrast = Ncontrast(x_dis, adj_label_batch, tau = args.tau)
    loss_train = loss_train_class + loss_Ncontrast * args.alpha
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    return loss_train, acc_train

def trainVAE():
    features_batch, adj_label_batch = get_batch(batch_size=args.batch_size)
    modelVAE.train()
    optimizerVAE.zero_grad()
    recon_x1, mu1, logvar1, output1, _, _ = modelVAE(features_batch)
    recon_x2, mu2, logvar2, output2, _, _ = modelVAE(features_batch)
    loss_train = vae_loss(recon_x1, features_batch, mu1, logvar1) + vae_loss(recon_x2, features_batch, mu2, logvar2)
    loss_total = loss_train
    loss_total.backward()
    optimizerVAE.step()
    return loss_total

def trainClassifier():
    features_batch, adj_label_batch = get_batch(batch_size=args.batch_size)
    modelVAE.train()
    optimizerClassifier.zero_grad()
    _, _, _, output1, x_dis1, z1 = modelVAE(features_batch)
    _, _, _, output2, x_dis2, z2 = modelVAE(features_batch)
    loss_Ncontrast = Ncontrast(x_dis1, adj_label_batch, tau=args.tau) + Ncontrast(x_dis2, adj_label_batch, tau=args.tau)
    xy_dis=get_contrastive_dis(z1,z2)
    loss_Ncontrast2 = Ncontrast2(xy_dis, tau=1)
    loss_class = F.nll_loss(output1[idx_train], labels[idx_train]) + F.nll_loss(output2[idx_train], labels[idx_train])
    # print("nll loss:",loss_class)
    loss_total = loss_class + loss_Ncontrast* args.alpha + loss_Ncontrast2* args.alpha*2
    acc_train = accuracy(output1[idx_train], labels[idx_train]) + accuracy(output2[idx_train], labels[idx_train])
    acc_train = acc_train/2
    loss_total.backward()
    optimizerClassifier.step()
    return loss_total, acc_train

def trainMLP():
    features_batch, adj_label_batch = get_batch(batch_size=args.batch_size)
    modelMLP.train()
    optimizerMLP.zero_grad()
    output = modelMLP(features_batch)
    loss_class = F.nll_loss(output[idx_train], labels[idx_train])
    # print("nll loss:",loss_class)
    loss_total = loss_class
    loss_total.backward()
    optimizerMLP.step()
    return loss_total

def test(model):
    model.eval()
    output = model(features)
    #here to change, no more data augmentation here
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    return acc_test, acc_val

best_accu = 0
best_val_acc = 0
print('\n'+'training configs', args)
start = time.time()
writer_train = SummaryWriter("runs/exp3/train")
writer_test = SummaryWriter("runs/exp3/test")

def traintestGraphMLP(best_val_acc):
    for epoch in tqdm(range(args.epochs)):
        #default epoch 400
        loss, acc = train()
        tmp_test_acc, val_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        writer_train.add_scalar("Loss/train", loss, epoch)
        writer_train.add_scalar("Accuracy", acc, epoch)
        writer_test.add_scalar("Accuracy/valid", val_acc, epoch)
        writer_test.add_scalar("Accuracy", tmp_test_acc, epoch)

        if epoch % 200 == 0 and epoch != 0:
            torch.save(model.state_dict(), args.result_dir_name + '/' + args.data + '_epoch_' + str(epoch + 1) + '.pkl')
        
## train VAE
def traintestVAE(best_val_acc):
    for epoch in tqdm(range(2*args.epochs)):
        #default epoch 400
        loss = trainVAE()
        tmp_test_acc, val_acc = test(modelVAE)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        if epoch % 200 == 0 and epoch != 0:
            torch.save(model.state_dict(), args.result_dir_name + '/' + args.data + '_epoch_' + str(epoch + 1) + '.pkl')
    print('best validate accuracy is:', best_val_acc.item())
    print('test accuracy is:', test_acc.item())

def traintestMLP(best_val_acc):
    #use MLP to compare our result
    for epoch in tqdm(range(args.epochs)):
        loss = trainMLP()
        tmp_test_acc, val_acc = test(modelMLP)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        if epoch % 200 == 0 and epoch != 0:
            torch.save(model.state_dict(), args.result_dir_name + '/' + args.data + '_epoch_' + str(epoch + 1) + '.pkl')
    print('best validate accuracy is:', best_val_acc.item())
    print('test accuracy is:', test_acc.item())

## freeze layer parameter
# unfreeze_layers = ['classifier3']
# for name, param in modelVAE.named_parameters():
#     param.requires_grad = False
#     for ele in unfreeze_layers:
#         if ele in name:
#             param.requires_grad = True
#             break

def traintestClassifier(best_val_acc):
    for epoch in tqdm(range(args.epochs)):
        #default epoch 400
        loss, acc = trainClassifier()
        tmp_test_acc, val_acc = test(modelVAE)
        # print(tmp_test_acc)
        # print(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        writer_train.add_scalar("Loss/train", loss, epoch)
        writer_train.add_scalar("Acc/train", acc, epoch)
        writer_test.add_scalar("Acc/test", tmp_test_acc, epoch)
        if epoch % 200 == 0 and epoch != 0:
            print(tmp_test_acc)
            torch.save(model.state_dict(), args.result_dir_name + '/' + args.data + '_epoch_' + str(epoch + 1) + '.pkl')
    print('best validate accuracy is:', best_val_acc.item())
    print('test accuracy is:', test_acc.item())


traintestVAE(best_val_acc)
traintestClassifier(best_val_acc)
writer_train.close()
writer_test.close()
end = time.time()
print('time cost of ', args.epochs, 'epochs:', end-start, 's')

# log_file = open(r"log.txt", encoding="utf-8",mode="a+")
# with log_file as file_to_be_write:  
#     print('tau', 'order', \
#             'batch_size', 'hidden', \
#                 'alpha', 'lr', \
#                     'weight_decay', 'data', \
#                         'test_acc', file=file_to_be_write, sep=',')
#     print(args.tau, args.order, \
#          args.batch_size, args.hidden, \
#              args.alpha, args.lr, \
#                  args.weight_decay, args.data, \
#                      test_acc.item(), file=file_to_be_write, sep=',')


