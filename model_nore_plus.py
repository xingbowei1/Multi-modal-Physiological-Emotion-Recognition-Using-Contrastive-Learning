import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from utils import normalize_A, generate_cheby_adj
import pdb

class Chebynet(nn.Module):
    def __init__(self, xdim, K, num_out, dropout):
        super(Chebynet, self).__init__()

        self.K = K
        self.gc1 = nn.ModuleList()
        for i in range(K):
            self.gc1.append(GraphConvolution(xdim[2], num_out))

    def forward(self, x,L):
        adj = generate_cheby_adj(L, self.K)
        for i in range(len(self.gc1)):
            if i == 0:
                result1 = self.gc1[i](x, adj[i])
            else:
                result1 += self.gc1[i](x, adj[i])

        result = F.relu(result1)
        return result


class GNN_1C(nn.Module):
    def __init__(self, xdim, k_adj, dropout,num_classes_1):
        super(GNN_1C, self).__init__()

        self.K = k_adj
        self.layer1 = Chebynet(xdim, k_adj, 32, dropout)
        #self.layer2 = IAG([xdim[0], xdim[1], 32], k_adj, 64, dropout)
        #self.layer3 = Chebynet([xdim[0], xdim[1], 64], k_adj, 128, dropout)
        self.dropout1 = nn.Dropout2d(dropout)
        #self.dropout2 = nn.Dropout2d(dropout)
        self.BN1 = nn.BatchNorm1d(5)
        self.fc1 = nn.Linear(xdim[1] * 32, 256)
        self.fc2 = nn.Linear(256, num_classes_1)
        self.A = nn.Parameter(torch.FloatTensor(xdim[1], xdim[1]).cuda())
        nn.init.kaiming_normal_(self.A, mode='fan_in')

    def forward(self, x):
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)
        L = normalize_A(self.A)
        result = self.layer1(x, L)

        result = result.reshape(x.shape[0], -1)

        result = F.relu(self.fc1(result))
        result = self.fc2(result)

        return result


class GNN_1S(nn.Module):
    def __init__(self, xdim, k_adj, dropout,num_classes_1):
        super(GNN_1S, self).__init__()

        self.K = k_adj
        self.layer1 = Chebynet(xdim, k_adj, 32, dropout)
        # self.layer2 = IAG([xdim[0], xdim[1], 32], k_adj, 64, dropout)
        # self.layer3 = Chebynet([xdim[0], xdim[1], 64], k_adj, 128, dropout)
        self.dropout1 = nn.Dropout2d(dropout)
        # self.dropout2 = nn.Dropout2d(dropout)
        self.BN1 = nn.BatchNorm1d(5)
        self.fc1 = nn.Linear(xdim[1] * 32, 256)
        self.fc2 = nn.Linear(256, num_classes_1)
        self.A = nn.Parameter(torch.FloatTensor(xdim[1], xdim[1]).cuda())
        nn.init.kaiming_normal_(self.A, mode='fan_in')

    def forward(self, x):
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)
        L = normalize_A(self.A)
        result = self.layer1(x, L)

        result = result.reshape(x.shape[0], -1)

        result = F.relu(self.fc1(result))
        result = self.fc2(result)

        return result


class NeuralNet_2C(nn.Module):
    def __init__(self, input_size_2, hidden_size_2, num_classes_2):
        super(NeuralNet_2C, self).__init__()
        self.BN1 = nn.BatchNorm1d(input_size_2,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fc1 = nn.Linear(input_size_2, hidden_size_2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_2, num_classes_2)
        #self.A = nn.Parameter(torch.FloatTensor(xdim[1], xdim[1]).cuda())

    def forward(self, x):
        x = self.BN1(x)
        out = self.fc1(x)
        #out = normalize_A(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class NeuralNet_2S(nn.Module):
    def __init__(self, input_size_2, hidden_size_2, num_classes_2):
        super(NeuralNet_2S, self).__init__()
        self.BN1 = nn.BatchNorm1d(input_size_2,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fc1 = nn.Linear(input_size_2, hidden_size_2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_2, num_classes_2)
        #self.A = nn.Parameter(torch.FloatTensor(xdim[1], xdim[1]).cuda())

    def forward(self, x):
        x = self.BN1(x)
        out = self.fc1(x)
        #out = normalize_A(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class NeuralNet_Re1(nn.Module):
    def __init__(self, input_size_Re1, hidden_size_Re1, num_classes_Re1):
        super(NeuralNet_Re1, self).__init__()
        self.BN1 = nn.BatchNorm1d(input_size_Re1,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fc1 = nn.Linear(input_size_Re1, hidden_size_Re1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_Re1, num_classes_Re1)
        #self.A = nn.Parameter(torch.FloatTensor(xdim[1], xdim[1]).cuda())

    def forward(self, x):
        x = self.BN1(x)
        out = self.fc1(x)
        #out = normalize_A(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class NeuralNet_Re2(nn.Module):
    def __init__(self, input_size_Re2, hidden_size_Re2, num_classes_Re2):
        super(NeuralNet_Re2, self).__init__()
        self.BN1 = nn.BatchNorm1d(input_size_Re2,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fc1 = nn.Linear(input_size_Re2, hidden_size_Re2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_Re2, num_classes_Re2)
        #self.A = nn.Parameter(torch.FloatTensor(xdim[1], xdim[1]).cuda())

    def forward(self, x):
        x = self.BN1(x)
        out = self.fc1(x)
        #out = normalize_A(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class Attention1(nn.Module):
    def __init__(self, in_channel, dropout=0, bias=True):
        super(Attention1, self).__init__()
        self.fc = nn.Linear(in_channel, 2, bias=bias)

    def forward(self, x):

        att = F.softmax(self.fc(x), 1)
        att1 = att[:,0:1]
        att2 = att[:,1:2]
        #device = att.device
        #indices1 = torch.LongTensor([0])
        #indices2 = torch.LongTensor([1])
        #att1 = att.index_select(1, indices1).to(device)
        #att2 = att.index_select(1, indices2).to(device)
        # att_all=torch.sum(att,1)
        # #x=x/att_all
        return att1,att2

class Attention2(nn.Module):
    def __init__(self, in_channel, dropout=0, bias=True):
        super(Attention, self).__init__()
        self.fc = Linear(in_channel, 1, bias=bias)

    def forward(self, x):
        att = F.softmax(self.fc(x), 1)
        x = x * att
        # att_all=torch.sum(att,1)
        # #x=x/att_all
        return x


class Fusion(nn.Module):
    def __init__(self,  xdim_EEG, k_adj,dropout,num_classes_1,input_size_2, hidden_size_2, num_classes_2,inputsize1, inputsize2,hidden_size_Re1,hidden_size_Re2,nclass):
        super(Fusion, self).__init__()
        self.gc_1C = GNN_1C(xdim_EEG, k_adj,dropout,num_classes_1)
        self.gc_1S = GNN_1S(xdim_EEG, k_adj,dropout,num_classes_1)

        self.gc_2C = NeuralNet_2C(input_size_2, hidden_size_2, num_classes_2)
        self.gc_2S = NeuralNet_2S(input_size_2, hidden_size_2, num_classes_2)
        self.att = Attention1(num_classes_1 + 2 * num_classes_2, dropout=0, bias=True)

        self.fc1 = nn.Sequential(
            nn.Linear(inputsize1, inputsize2),
            nn.ReLU(inplace=True),
            nn.Linear(inputsize2, nclass))

    def forward(self, x1, x2):

        #x1 = x1.reshape(x1.shape[0], -1)

        out_1C = self.gc_1C(x1)
        out_1S = self.gc_1S(x1)

        x2 = x2.reshape(x2.shape[0], -1)
        out_2C = self.gc_2C(x2)
        out_2S = self.gc_2S(x2)

        '''

        F11 = torch.cat((out_1C,out_1S),dim=1)
        att_1C,att_1S = self.att_1(F11)

        F21 = torch.cat((out_2C,out_2S),dim=1)
        att_2C, att_2S = self.att_2(F21)

        o1C = out_1C * att_1C
        o1S = out_1S * att_1S

        o2C = out_2C * att_2C
        o2S = out_2S * att_2S

        F1 = torch.cat((o1C, o1S), dim=1)
        F2 = torch.cat((o2C, o2S), dim=1)
        '''

        #F1_Re = self.gc_Re1(F1)
        #F2_Re = self.gc_Re2

        FC1 = out_1C + out_2C
        FS1 = torch.cat((out_1S,out_2S),dim=1)

        x_Fusion2 = torch.cat((FC1, FS1), dim=1)

        att_1, att_2 = self.att(x_Fusion2)

        FC = FC1 * att_1
        FS = FS1 * att_2

        x_Fusion = torch.cat((FC, FS), dim=1)

        x_Fusion1 = F.normalize(self.fc1(x_Fusion), dim=1)

        return x1,x2,out_1C,out_1S,out_2C,out_2S,x_Fusion,x_Fusion1




'''
        self.fc3 = nn.Linear(input_size2, hidden_size2)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size2, hidden_size3)
        self.fc5 = nn.Linear(hidden_size3, nclass)
        self.fc6 = nn.Linear(num_classes2, num_classes2)
        self.fc7 = nn.Sequential(
            nn.Linear(input_size3, input_size3),
            nn.ReLU(inplace=True),
            nn.Linear(input_size3, num_classes2)
        )
        self.fc8 = nn.Sequential(
            nn.Linear(input_size2, input_size2),
            nn.ReLU(inplace=True),
            nn.Linear(input_size2, hidden_size2))
        self.fc9 = nn.Sequential(
            nn.Linear(310, 310),
            nn.ReLU(inplace=True),
            nn.Linear(310, num_classes2))
        self.fc10 = nn.Sequential(
            nn.Linear(31, 31),
            nn.ReLU(inplace=True),
            nn.Linear(31, num_classes2))
'''



'''
        out11 = F.normalize(self.fc7(out1), dim=1)
        #out12 = self.relu(out11)
        #out12 = self.fc6(out12)
        #out12 = F.normalize(self.fc9(out12), dim=1)

        out21 = F.normalize(self.fc7(out2), dim=1)
        #out22 = self.relu(out21)
        #out22 = self.fc6(out22)
        #out22 = F.normalize(self.fc9(out22), dim=1)

        input1 = x_EEG.reshape(x_EEG.shape[0], -1)
        input1 = F.normalize(self.fc9(input1), dim=1)

        input2 = x_EYE.reshape(x_EYE.shape[0], -1)
        input2 = F.normalize(self.fc10(input2), dim=1)
'''


'''
        x_Fusion2 = self.relu(x_Fusion1)
        x_Fusion2 = self.fc4(x_Fusion2)
        x_Fusion2 = self.relu(x_Fusion2)
        x_Fusion3 = self.fc5(x_Fusion2)
'''













'''

class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x

class SupConGNN(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='GNN', head = 'mlp', feat_dim = 128):
        super(SupConGNN,self).__init__()
        model_fun,dim_in = model_dict[]
        self.encoder = model_fun()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLu(inplace = Ture),
                nn.Linear(dim_in,feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self,x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat),dim=1)
        return feat

class SupCEGNN(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='GNN', num_classes=10):
        super(SupCEGNN, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='GNN', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)
        
'''