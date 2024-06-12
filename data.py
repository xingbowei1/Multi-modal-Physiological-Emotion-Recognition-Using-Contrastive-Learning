import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as scio


def load_data():
    xtrain = np.random.random(size=[1000, 62, 5])
    xtest = np.random.random(size=[200, 62, 5])
    ytrain = np.random.randint(3, size=[1000])
    ytest = np.random.randint(3, size=[200])
    return xtrain, xtest, ytrain, ytest





def load_data_EEG_MPED(num=2, flag=4, flag_band=5, flag_class=2, channel_t=0):
    # emotion list:[neural,joy,funny,angry,fear,disgust,sadness],[positive,neutral,negative]
    # num:refer to protocol
    # flag:refer to features
    # flag_band:refer to band

    with open("/home/xingbowei/MPED1/MPED.txt", "r") as f:
        name = np.array([])
        for line in f.readlines():
            line = line.strip('\n')
            name = np.append(name, line)

    features = ['HHS', 'Hjorth', 'HOC', 'PSD', 'STFT']
    band = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'all']

    if num == 1:
        emotion = {1: [2, 8, 16, 21],  # neutral
                   2: [1, 9, 19, 20],  # joy
                   3: [4, 10, 26, 28],  # funny
                   4: [3, 13, 24, 25],  # angry
                   5: [5, 14, 17, 27],  # fear
                   6: [6, 11, 18, 23],  # disgust
                   7: [7, 12, 22, 29]}  # sadness
        class_index = [[2, 1, 4],
                       [2, 1, 5],
                       [2, 1, 6],
                       [2, 1, 7],
                       [3, 1, 4],
                       [3, 1, 5],
                       [3, 1, 6],
                       [3, 1, 7]]
        Pos, N, Neg = class_index[flag_class]
        train_id = emotion[Pos][:3] + emotion[N][:3] + emotion[Neg][:3]
        test_id = emotion[Pos][3:] + emotion[N][3:] + emotion[Neg][3:]

        feature_data_train = np.zeros(shape=(1, 23), dtype=object)
        feature_data_test = np.zeros(shape=(1, 23), dtype=object)

        for i in range(name.shape[0]):
            path = '/home/xingbowei/MPED1/EEG_feature/' + \
                features[flag] + '/' + name[i] + '.mat'  # F:/Desktop/毕设/
            data = scio.loadmat(path)

            if flag == 1 or flag == 2:
                data = data[band[flag_band]]
            elif flag == 3 or flag == 4:
                data = data[features[flag]]
            else:
                data = data['hhs_A']

            data_test = data[:, test_id]
            data_train = data[:, train_id]

            p_data_train = np.concatenate((data_train[0, :]), axis=1)
            p_data_train = p_data_train.transpose([1, 0, 2])
            feature_data_train[0, i] = p_data_train

            p_data_test = np.concatenate((data_test[0, :]), axis=1)
            p_data_test = p_data_test.transpose([1, 0, 2])
            feature_data_test[0, i] = p_data_test

            if i == 0:
                sh_train = [data_train[0, j].shape[1]
                            for j in range(data_train.shape[1])]
                sh_test = [data_test[0, j].shape[1]
                           for j in range(data_test.shape[1])]

    else:
        test_id = [20, 21, 23, 25, 27, 28, 29]

        feature_data_train = np.zeros(shape=(1, 23), dtype=object)
        feature_data_test = np.zeros(shape=(1, 23), dtype=object)

        for i in range(name.shape[0]):  # name.shape[0]
            path = '/home/xingbowei/MPED1/EEG_feature/' + \
                features[flag] + '/' + name[i] + '.mat'  # F:/Desktop/毕设/
            data = scio.loadmat(path)

            if flag == 1 or flag == 2:
                data = data[band[flag_band]]
            elif flag == 3 or flag == 4:
                data = data[features[flag]]
            else:
                data = data['hhs_A']

            data_test = data[:, test_id]
            data_train = np.delete(data, test_id + [0, 15], 1)

            p_data_train = np.concatenate((data_train[0, :]), axis=1)
            p_data_train = p_data_train.transpose([1, 0, 2])

            # p_data_train=p_data_train[:,channel[channel_t],:]
            feature_data_train[0, i] = p_data_train

            p_data_test = np.concatenate((data_test[0, :]), axis=1)
            p_data_test = p_data_test.transpose([1, 0, 2])

            # p_data_test=p_data_test[:,channel[channel_t],:]
            feature_data_test[0, i] = p_data_test

            if i == 0:
                sh_train = [data_train[0, j].shape[1]
                            for j in range(data_train.shape[1])]
                sh_test = [data_test[0, j].shape[1]
                           for j in range(data_test.shape[1])]

    # 协议一、二
    if num == 2 or num == 1:
        path = '/home/xingbowei/MPED1/label_3.mat'  # F:/Desktop/毕设/
        label = scio.loadmat(path)
        label = label['label_3']
        label = label.reshape(30, 1)

    # 协议三
    else:
        path = '/home/xingbowei/MPED1/label.mat'  # F:/Desktop/毕设/
        label = scio.loadmat(path)
        label = label['label']
        label = label.reshape(30, 1)

    if num == 1:
        label_train = label[train_id]
    else:
        label_train = np.delete(label, test_id + [0, 15], 0)
    label_train = label_train - 1
    label_train = label_train.repeat(sh_train, axis=0).flatten()

    label_test = label[test_id]
    label_test = label_test - 1
    label_test = label_test.repeat(sh_test, axis=0).flatten()

    return feature_data_train, feature_data_test, label_train, label_test


def load_data_3_MPED(num=2, flag=2, flag_class=2, channel_t=0):
    # emotion list:[neural,joy,funny,angry,fear,disgust,sadness],[positive,neutral,negative]
    # num:refer to protocol
    # flag:refer to features
    # flag_band:refer to band

    with open("/home/xingbowei/MPED1/MPED.txt", "r") as f:
        name = np.array([])
        for line in f.readlines():
            line = line.strip('\n')
            name = np.append(name, line)

    features = ['ECG_Feature', 'GSR_Feature', 'RSP_Feature']


    if num == 1:
        emotion = {1: [2, 8, 16, 21],  # neutral
                   2: [1, 9, 19, 20],  # joy
                   3: [4, 10, 26, 28],  # funny
                   4: [3, 13, 24, 25],  # angry
                   5: [5, 14, 17, 27],  # fear
                   6: [6, 11, 18, 23],  # disgust
                   7: [7, 12, 22, 29]}  # sadness
        class_index = [[2, 1, 4],
                       [2, 1, 5],
                       [2, 1, 6],
                       [2, 1, 7],
                       [3, 1, 4],
                       [3, 1, 5],
                       [3, 1, 6],
                       [3, 1, 7]]
        Pos, N, Neg = class_index[flag_class]
        train_id = emotion[Pos][:3] + emotion[N][:3] + emotion[Neg][:3]
        test_id = emotion[Pos][3:] + emotion[N][3:] + emotion[Neg][3:]

        feature_data1_train = np.zeros(shape=(1, 23), dtype=object)
        feature_data1_test = np.zeros(shape=(1, 23), dtype=object)

        feature_data2_train = np.zeros(shape=(1, 23), dtype=object)
        feature_data2_test = np.zeros(shape=(1, 23), dtype=object)

        feature_data3_train = np.zeros(shape=(1, 23), dtype=object)
        feature_data3_test = np.zeros(shape=(1, 23), dtype=object)

        for i in range(name.shape[0]):
            path = '/home/xingbowei/MPED1/GSR_RSP_ECG_features/' + \
                 name[i] + '.mat'  # F:/Desktop/毕设/
            data = scio.loadmat(path)

            data1 = data[features[0]]
            data2 = data[features[1]]
            data3 = data[features[2]]

            '''

            if flag == 1 or flag == 2:
                data = data[band[flag_band]]
            elif flag == 3 or flag == 4:
                data = data[features[flag]]
            else:
                data = data['hhs_A']
            '''

            data1_test = data1[:, test_id]
            data1_train = data1[:, train_id]

            p_data1_train = np.concatenate((data1_train[0, :]), axis=1)
            p_data1_train = p_data1_train.transpose([1, 0])
            feature_data1_train[0, i] = p_data1_train

            p_data1_test = np.concatenate((data1_test[0, :]), axis=1)
            p_data1_test = p_data1_test.transpose([1, 0])
            feature_data1_test[0, i] = p_data1_test

            if i == 0:
                sh1_train = [data1_train[0, j].shape[1]
                            for j in range(data1_train.shape[1])]
                sh1_test = [data1_test[0, j].shape[1]
                           for j in range(data1_test.shape[1])]

            data2_test = data2[:, test_id]
            data2_train = data2[:, train_id]

            p_data2_train = np.concatenate((data2_train[0, :]), axis=1)
            p_data2_train = p_data2_train.transpose([1, 0])
            feature_data2_train[0, i] = p_data2_train

            p_data2_test = np.concatenate((data2_test[0, :]), axis=1)
            p_data2_test = p_data2_test.transpose([1, 0])
            feature_data2_test[0, i] = p_data2_test

            if i == 0:
                sh2_train = [data2_train[0, j].shape[1]
                             for j in range(data2_train.shape[1])]
                sh2_test = [data2_test[0, j].shape[1]
                            for j in range(data2_test.shape[1])]

            data3_test = data3[:, test_id]
            data3_train = data3[:, train_id]

            p_data3_train = np.concatenate((data3_train[0, :]), axis=1)
            p_data3_train = p_data3_train.transpose([1, 0])
            feature_data3_train[0, i] = p_data3_train

            p_data3_test = np.concatenate((data3_test[0, :]), axis=1)
            p_data3_test = p_data3_test.transpose([1, 0])
            feature_data3_test[0, i] = p_data3_test

            if i == 0:
                sh3_train = [data3_train[0, j].shape[1]
                             for j in range(data3_train.shape[1])]
                sh3_test = [data3_test[0, j].shape[1]
                            for j in range(data3_test.shape[1])]


    else:
        test_id = [20, 21, 23, 25, 27, 28, 29]

        feature_data1_train = np.zeros(shape=(1, 23), dtype=object)
        feature_data1_test = np.zeros(shape=(1, 23), dtype=object)

        feature_data2_train = np.zeros(shape=(1, 23), dtype=object)
        feature_data2_test = np.zeros(shape=(1, 23), dtype=object)

        feature_data3_train = np.zeros(shape=(1, 23), dtype=object)
        feature_data3_test = np.zeros(shape=(1, 23), dtype=object)

        for i in range(name.shape[0]):  # name.shape[0]
            path = '/home/xingbowei/MPED1/GSR_RSP_ECG_features/' + \
                 name[i] + '.mat'  # F:/Desktop/毕设/
            data = scio.loadmat(path)

            data1 = data[features[0]]
            data2 = data[features[1]]
            data3 = data[features[2]]

            data1_test = data1[:, test_id]
            data1_train = np.delete(data1, test_id + [0, 15], 1)

            data2_test = data2[:, test_id]
            data2_train = np.delete(data2, test_id + [0, 15], 1)

            data3_test = data3[:, test_id]
            data3_train = np.delete(data3, test_id + [0, 15], 1)


            p_data1_train = np.concatenate((data1_train[0, :]), axis=1)
            p_data1_train = p_data1_train.transpose([1, 0])

            p_data2_train = np.concatenate((data2_train[0, :]), axis=1)
            p_data2_train = p_data2_train.transpose([1, 0])

            p_data3_train = np.concatenate((data3_train[0, :]), axis=1)
            p_data3_train = p_data3_train.transpose([1, 0])

            # p_data_train=p_data_train[:,channel[channel_t],:]
            feature_data1_train[0, i] = p_data1_train
            feature_data2_train[0, i] = p_data2_train
            feature_data3_train[0, i] = p_data3_train

            p_data1_test = np.concatenate((data1_test[0, :]), axis=1)
            p_data1_test = p_data1_test.transpose([1, 0])

            p_data2_test = np.concatenate((data2_test[0, :]), axis=1)
            p_data2_test = p_data2_test.transpose([1, 0])

            p_data3_test = np.concatenate((data3_test[0, :]), axis=1)
            p_data3_test = p_data3_test.transpose([1, 0])

            # p_data_test=p_data_test[:,channel[channel_t],:]
            feature_data1_test[0, i] = p_data1_test
            feature_data2_test[0, i] = p_data2_test
            feature_data3_test[0, i] = p_data3_test

            if i == 0:
                sh1_train = [data1_train[0, j].shape[1]
                            for j in range(data1_train.shape[1])]
                sh1_test = [data1_test[0, j].shape[1]
                           for j in range(data1_test.shape[1])]
                sh2_train = [data2_train[0, j].shape[1]
                             for j in range(data2_train.shape[1])]
                sh2_test = [data2_test[0, j].shape[1]
                            for j in range(data2_test.shape[1])]
                sh3_train = [data3_train[0, j].shape[1]
                             for j in range(data3_train.shape[1])]
                sh3_test = [data3_test[0, j].shape[1]
                            for j in range(data3_test.shape[1])]
    return feature_data1_train, feature_data1_test, feature_data2_train, feature_data2_test, feature_data3_train, feature_data3_test
'''
    # 协议一、二
    if num == 2 or num == 1:
        path = '/mnt/xingbowei/MPED1/label_3.mat'  # F:/Desktop/毕设/
        label = scio.loadmat(path)
        label = label['label_3']
        label = label.reshape(30, 1)

    # 协议三
    else:
        path = '/mnt/xingbowei/MPED1/label.mat'  # F:/Desktop/毕设/
        label = scio.loadmat(path)
        label = label['label']
        label = label.reshape(30, 1)

    if num == 1:
        label_train = label[train_id]
    else:
        label_train = np.delete(label, test_id + [0, 15], 0)
    label_train = label_train - 1
    label_train = label_train.repeat(sh_train, axis=0).flatten()

    label_test = label[test_id]
    label_test = label_test - 1
    label_test = label_test.repeat(sh_test, axis=0).flatten()
'''



