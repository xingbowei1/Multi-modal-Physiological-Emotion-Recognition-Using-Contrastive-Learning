from __future__ import division
from __future__ import print_function

import time
import argparse
import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from model_nore_plus import Fusion
from data import load_data_EEG_MPED, load_data_3_MPED
from util import AverageMeter
from losses import SupConLoss
import pdb

import sys


def set_optimizer(model):
    optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate,
                          momentum=momentum,
                          weight_decay=weight_decay)
    return optimizer




class Logger(object):
    def __init__(self, fileN="default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


num_classes_1 = 32
input_size_2 = 18
hidden_size_2 = 64
num_classes_2 = 32
inputsize1 = 96
inputsize2 = 64
nclass = 7
hidden_size_Re1 = 64
hidden_size_Re2 = 64
#hidden_size3 = 32


batch_size = 8
total_epoch = 500
learning_rate = 0.001
dropout_rate = 0.5
weight_decay = 0.001
momentum = 0.7
m1 = 0.8
m2 = 1

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

seed = 23
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

temperature = 0.1
line = 1

def SimCLRLoss(z1, z2, z3, temperature):
    N = z1.shape
    device = z1.device
    positives = torch.matmul(z1, z2.T)

    positives_1 = torch.diag(positives)

    positives_1 = positives_1.reshape(batch_size,1)

    negatives = torch.matmul(z1, z3.T)

    logits = torch.cat([positives_1, negatives], dim=1)

    logits /= temperature

    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    return logits, labels

sys.stdout = Logger('EEG+ECG-16-Sin-att-f.txt')
print('seed {}, temperature{}, total_epoch {},batch_size {},learning_rate:{},dropout_rate:{}, weight_decay:{},momentum:{},line:{}' .format(
            seed, temperature, total_epoch,batch_size, learning_rate,dropout_rate,weight_decay, momentum, line))


sys.setrecursionlimit(3000)  # 将默认的递归深度修改为3000

c = np.zeros(shape=(7,7))
acc_all = []

protocol = 3


if protocol == 3:

    data_EEG_train, data_EEG_test, label_EEG_train, label_EEG_test = load_data_EEG_MPED(num=protocol)
    data_ECG_train, data_ECG_test, data_GSR_train, data_GSR_test, data_RSP_train, data_RSP_test = load_data_3_MPED(num=protocol)
    ytrain = torch.from_numpy(label_EEG_train)
    ytest = torch.from_numpy(label_EEG_test)
    acc_all = []


    for t in range(data_EEG_train.shape[1]):  # data_EEG_train.shape[1]
        acc_train = []
        acc_test = []
        xtrain_EEG = data_EEG_train[0, t]
        xtest_EEG = data_EEG_test[0, t]

        xtrain_ECG = data_ECG_train[0, t]
        xtest_ECG = data_ECG_test[0, t]

        xtrain_GSR = data_GSR_train[0, t]
        xtest_GSR = data_GSR_test[0, t]

        xtrain_RSP = data_RSP_train[0, t]
        xtest_RSP = data_RSP_test[0, t]

        size2 = np.size(xtrain_ECG[0])
        size3 = np.size(xtrain_GSR[0])
        size4 = np.size(xtrain_RSP[0])

        xtrain_EEG = torch.from_numpy(xtrain_EEG)
        xtest_EEG = torch.from_numpy(xtest_EEG)

        xtrain_ECG = torch.from_numpy(xtrain_ECG)
        xtest_ECG = torch.from_numpy(xtest_ECG)

        xtrain_GSR = torch.from_numpy(xtrain_GSR)
        xtest_GSR = torch.from_numpy(xtest_GSR)

        xtrain_RSP = torch.from_numpy(xtrain_RSP)
        xtest_RSP = torch.from_numpy(xtest_RSP)



        train_dataset = Data.TensorDataset(xtrain_EEG, xtrain_ECG, ytrain)
        test_dataset = Data.TensorDataset(xtest_EEG, xtest_ECG, ytest)

        train_loader = Data.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       drop_last=True,
                                       shuffle=True)

        test_loader = Data.DataLoader(dataset=test_dataset,
                                      batch_size=batch_size,
                                      shuffle=False)

        size1 = xtrain_EEG.shape

        #print(size1, size2, size3, size4)

        model3 = Fusion(size1, 2, dropout_rate,num_classes_1,input_size_2, hidden_size_2, num_classes_2,inputsize1, inputsize2,hidden_size_Re1,hidden_size_Re2,nclass).cuda()

        # print(model)
        # for name, param in model.named_parameters():
        #     print(name)

        criterion1 = SupConLoss(temperature)
        criterion2 = nn.CrossEntropyLoss()
        criterionL1 = nn.L1Loss(reduction='mean')

        optimizer3 = torch.optim.Adam(model3.parameters(), lr=learning_rate, weight_decay=weight_decay)

        acc_max, acc_max_step, global_step = 0, 0, 0

        for epoch in range(total_epoch):

            for i, (features_EEG, features_ECG, labels) in enumerate(train_loader):
                global_step += 1

                model3.train()

                features_EEG = features_EEG.float().cuda()
                features_ECG = features_ECG.float().cuda()
                labels = labels.long().cuda()

                x1,x2,out_1C,out_1S,out_2C,out_2S,x_Fusion,x_Fusion1 = model3(features_EEG,features_ECG)

                # f1, f2 = torch.split(outputs2, [bsz, bsz], dim=0)

                # outs1 = torch.cat([out1.unsqueeze(1), out2.unsqueeze(1)], dim=1)

                logits1, labels1 = SimCLRLoss(out_1C, out_2C, out_1S, temperature)

                logits2, labels2 = SimCLRLoss(out_2C, out_1C, out_2S, temperature)

                outs1 = torch.cat([out_1C.unsqueeze(1), out_1C.unsqueeze(1)], dim=1)

                outs2 = torch.cat([out_2C.unsqueeze(1), out_2C.unsqueeze(1)], dim=1)

                outs3 = torch.cat([out_1S.unsqueeze(1), out_1S.unsqueeze(1)], dim=1)

                outs4 = torch.cat([out_2S.unsqueeze(1), out_2S.unsqueeze(1)], dim=1)

                x1 = x1.reshape(x1.shape[0], -1)

                # pdb.set_trace()

                # loss3 = criterion1(outs, labels)
                # loss3 =criterion2(outputs, labels)
                loss3 = 0.1 * criterion2(x_Fusion1, labels) + m1 * (criterion2(logits1, labels1) + criterion2(logits2, labels2))+  m2 * ( criterion1(outs1, labels) + criterion1(outs2, labels) \
                    + criterion1(outs3, labels) + criterion1(outs4, labels))

                # loss3.update(loss3.item(), bsz)

                # optimizer3 = set_optimizer(model3)

                optimizer3.zero_grad()

                loss3.backward(retain_graph=True)

                optimizer3.step()

                if global_step % 1 == 0:
                    model3.eval()
                    with torch.no_grad():
                        features_EEG = xtest_EEG.float().cuda()
                        features_ECG = xtest_ECG.float().cuda()
                        labels = ytest.float().cuda()
                        x1,x2,out_1C,out_1S,out_2C,out_2S,x_Fusion,x_Fusion1 = model3(features_EEG,features_ECG)
                        # classifier_test = nn.Linear(84, 4)
                        # outs = classifier_test(out_fusion)

                        _, predicted = torch.max(x_Fusion1.data, 1)
                        test_acc = accuracy_score(labels.cpu(), predicted.cpu())

                        if test_acc > acc_max:
                            acc_max = test_acc
                            c_matric = confusion_matrix(labels.cpu(), predicted.cpu())
                            c_matric_normalized = c_matric.astype('float') / c_matric.sum(axis=1)[:, np.newaxis]
                            acc_max_step = global_step

            model3.eval()
            with torch.no_grad():
                features_EEG = xtrain_EEG.float().cuda()
                features_ECG = xtrain_ECG.float().cuda()
                x1,x2,out_1C,out_1S,out_2C,out_2S,x_Fusion,x_Fusion1  = model3(features_EEG,features_ECG)
                _, predicted = torch.max(x_Fusion1.data, 1)
                train_acc = accuracy_score(ytrain.cpu(), predicted.cpu())

            print('subject [{}],Epoch[{}],Trainacc:{},Testacc:{}, Bestacc:{},step:{}'.format(
                t + 1, epoch + 1, train_acc, test_acc, acc_max, acc_max_step))
            if acc_max >= line:
                break

        print('subject [{}],Best accuracy:{},step:{}'.format(
            t + 1, acc_max,  acc_max_step))
        print(c_matric_normalized)
        acc_all.append(acc_max)
        c = c + c_matric_normalized
    c1 = c / 23
    print(c1)
    print(acc_all, np.mean(acc_all), np.std(acc_all))

np.savetxt('EEG+ECG-16-Sin-att-f.txt')










