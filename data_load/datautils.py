import os
import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, LabelEncoder
from einops import rearrange
from data_load import data_loader
import natsort
from sklearn.model_selection import train_test_split


def load(dataset, max_dim):
    train_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TEST.arff')[0]

    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([d.tolist() for d in t_data])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)

    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)
    train_X = train_X.transpose(0, 2, 1)
    test_X = test_X.transpose(0, 2, 1)
    if dataset == 'FaceDetection':
        # FaceDetection's length: 62->63
        zero_pad = np.zeros((train_X.shape[0], train_X.shape[1], 1))
        train_X = np.concatenate((train_X, zero_pad), axis=2)
        zero_pad = np.zeros((test_X.shape[0], test_X.shape[1], 1))
        test_X = np.concatenate((test_X, zero_pad), axis=2)

    elif dataset == 'JapaneseVowels':
        # FaceDetection's length: 29->30
        zero_pad = np.zeros((train_X.shape[0], train_X.shape[1], 1))
        train_X = np.concatenate((train_X, zero_pad), axis=2)
        zero_pad = np.zeros((test_X.shape[0], test_X.shape[1], 1))
        test_X = np.concatenate((test_X, zero_pad), axis=2)

    elif dataset == 'Cricket':
        # Cricket's length: 1197->1200
        zero_pad = np.zeros((train_X.shape[0], train_X.shape[1], 3))
        train_X = np.concatenate((train_X, zero_pad), axis=2)
        zero_pad = np.zeros((test_X.shape[0], test_X.shape[1], 3))
        test_X = np.concatenate((test_X, zero_pad), axis=2)
    elif dataset == 'EigenWorms':
        # EigenWorms' length: 17984->17985
        # zero_pad = np.zeros((train_X.shape[0], train_X.shape[1], 1))
        # train_X = np.concatenate((train_X, zero_pad), axis=2)
        # zero_pad = np.zeros((test_X.shape[0], test_X.shape[1], 1))
        # test_X = np.concatenate((test_X, zero_pad), axis=2)
        # EigenWorms' length : 17984-> 17982
        train_X = train_X[:, :, :len(train_X[0][0]) - 2]
        test_X = test_X[:, :, :len(test_X[0][0]) - 2]

    elif dataset == 'Epilepsy':
        # Epilepsy's length: 206->208
        zero_pad = np.zeros((train_X.shape[0], train_X.shape[1], 2))
        train_X = np.concatenate((train_X, zero_pad), axis=2)
        zero_pad = np.zeros((test_X.shape[0], test_X.shape[1], 2))
        test_X = np.concatenate((test_X, zero_pad), axis=2)
    elif dataset == 'EthanolConcentration':
        # EthanolConcentration's length: 1751->1755
        zero_pad = np.zeros((train_X.shape[0], train_X.shape[1], 4))
        train_X = np.concatenate((train_X, zero_pad), axis=2)
        zero_pad = np.zeros((test_X.shape[0], test_X.shape[1], 4))
        test_X = np.concatenate((test_X, zero_pad), axis=2)
    elif dataset == 'NATOPS':
        # NATOPS' length: 51->54
        zero_pad = np.zeros((train_X.shape[0], train_X.shape[1], 3))
        train_X = np.concatenate((train_X, zero_pad), axis=2)
        zero_pad = np.zeros((test_X.shape[0], test_X.shape[1], 3))
        test_X = np.concatenate((test_X, zero_pad), axis=2)
    elif dataset == 'PhonemeSpectra':
        # PhonemeSpectra's length: 217->220
        zero_pad = np.zeros((train_X.shape[0], train_X.shape[1], 3))
        train_X = np.concatenate((train_X, zero_pad), axis=2)
        zero_pad = np.zeros((test_X.shape[0], test_X.shape[1], 3))
        test_X = np.concatenate((test_X, zero_pad), axis=2)
    elif dataset == 'SpokenArabicDigits':
        # SpokenArabicDigits's length: 93->96
        zero_pad = np.zeros((train_X.shape[0], train_X.shape[1], 3))
        train_X = np.concatenate((train_X, zero_pad), axis=2)
        zero_pad = np.zeros((test_X.shape[0], test_X.shape[1], 3))
        test_X = np.concatenate((test_X, zero_pad), axis=2)
    elif dataset == 'ERing':
        # ERing's length: 65->66
        zero_pad = np.zeros((train_X.shape[0], train_X.shape[1], 1))
        train_X = np.concatenate((train_X, zero_pad), axis=2)
        zero_pad = np.zeros((test_X.shape[0], test_X.shape[1], 1))
        test_X = np.concatenate((test_X, zero_pad), axis=2)
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    le = LabelEncoder()
    le.fit(train_y)
    train_y = le.transform(train_y)
    test_y = le.transform(test_y)

    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2, random_state=20,
                                                      stratify=train_y)

    train_dataset = data_loader.My_Multivariate_Data(train_X, train_y)
    val_dataset = data_loader.My_Multivariate_Data(val_X, val_y)
    test_dataset = data_loader.My_Multivariate_Data(test_X, test_y)

    return train_dataset, val_dataset, test_dataset


def load_UEA(dataset, max_dim):
    train_X = np.load(f'datasets/UEA/{dataset}/{dataset}_train_x.npy')
    train_y = np.load(f'datasets/UEA/{dataset}/{dataset}_train_label.npy')
    test_X = np.load(f'datasets/UEA/{dataset}/{dataset}_test_x.npy')
    test_y = np.load(f'datasets/UEA/{dataset}/{dataset}_test_label.npy')

    train_X = np.nan_to_num(train_X)
    test_X = np.nan_to_num(test_X)

    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    if dataset == 'FaceDetection':
        # FaceDetection's length: 62->63
        zero_pad = np.zeros((train_X.shape[0], train_X.shape[1], 1))
        train_X = np.concatenate((train_X, zero_pad), axis=2)
        zero_pad = np.zeros((test_X.shape[0], test_X.shape[1], 1))
        test_X = np.concatenate((test_X, zero_pad), axis=2)

    elif dataset == 'JapaneseVowels':
        # FaceDetection's length: 29->30
        zero_pad = np.zeros((train_X.shape[0], train_X.shape[1], 1))
        train_X = np.concatenate((train_X, zero_pad), axis=2)
        zero_pad = np.zeros((test_X.shape[0], test_X.shape[1], 1))
        test_X = np.concatenate((test_X, zero_pad), axis=2)

    dim_num = train_X.shape[1]
    merge_dim = False
    if dim_num > max_dim:
        new_dim_num = int(dim_num / max_dim) + 1 if int(dim_num % max_dim != 0) else int(dim_num / max_dim)
        pad_dim = max_dim - dim_num % max_dim if int(dim_num % max_dim != 0) else 0
        if pad_dim != 0:
            zero_pad = np.zeros((train_X.shape[0], pad_dim, train_X.shape[2]))
            train_X = np.concatenate((train_X, zero_pad), axis=1)
            zero_pad = np.zeros((test_X.shape[0], pad_dim, test_X.shape[2]))
            test_X = np.concatenate((test_X, zero_pad), axis=1)

        train_X = rearrange(train_X, 'n (d1 d2) f -> n d1 (d2 f)', d1=max_dim)
        test_X = rearrange(test_X, 'n (d1 d2) f -> n d1 (d2 f)', d1=max_dim)
        merge_dim = True

    le = LabelEncoder()
    le.fit(train_y)
    train_y = le.transform(train_y)
    test_y = le.transform(test_y)

    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2, random_state=20,
                                                      stratify=train_y)
    if not merge_dim:
        train_dataset = data_loader.My_Multivariate_Data(train_X, train_y)
        val_dataset = data_loader.My_Multivariate_Data(val_X, val_y)
        test_dataset = data_loader.My_Multivariate_Data(test_X, test_y)
    else:
        train_dataset = data_loader.My_Multivariate_Data(train_X, train_y, new_dim_num)
        val_dataset = data_loader.My_Multivariate_Data(val_X, val_y, new_dim_num)
        test_dataset = data_loader.My_Multivariate_Data(test_X, test_y, new_dim_num)
    return train_dataset, val_dataset, test_dataset


def load_UEA_csv(dataset, max_dim):
    path = f'datasets/UEA_csv/{dataset}'
    i = 0
    j = 0
    files = os.listdir(path)
    files = natsort.natsorted(files)
    for c in files:
        if c.endswith('TRAIN.csv'):
            df = pd.read_csv(os.path.join(path, c), header=None)
            df = df.replace('?', np.nan).astype(np.float32)
            train = df.to_numpy()
            if i == 0:
                train_X = train[:, np.newaxis, :]
            else:
                train_X = np.concatenate((train_X, train[:, np.newaxis, :]), axis=1)
            i += 1
        elif c.endswith("TEST.csv"):
            df = pd.read_csv(os.path.join(path, c), header=None)
            df = df.replace('?', np.nan).astype(np.float32)
            test = df.to_numpy()
            if j == 0:
                test_X = test[:, np.newaxis, :]
            else:
                test_X = np.concatenate((test_X, test[:, np.newaxis, :]), axis=1)
            j += 1
        elif c.startswith('train'):
            train_y = pd.read_csv(os.path.join(path, c), header=None).to_numpy().squeeze(1)

        elif c.startswith('test'):
            test_y = pd.read_csv(os.path.join(path, c), header=None).to_numpy().squeeze(1)

    train_X = np.nan_to_num(train_X)
    test_X = np.nan_to_num(test_X)
    if dataset == 'Cricket':
        # Cricket's length: 1197->1200
        zero_pad = np.zeros((train_X.shape[0], train_X.shape[1], 3))
        train_X = np.concatenate((train_X, zero_pad), axis=2)
        zero_pad = np.zeros((test_X.shape[0], test_X.shape[1], 3))
        test_X = np.concatenate((test_X, zero_pad), axis=2)
    elif dataset == 'EigenWorms':
        # EigenWorms' length : 17984-> 17982
        train_X = train_X[:, :, :len(train_X[0][0]) - 2]
        test_X = test_X[:, :, :len(test_X[0][0]) - 2]

    elif dataset == 'Epilepsy':
        # Epilepsy's length: 206->208
        zero_pad = np.zeros((train_X.shape[0], train_X.shape[1], 2))
        train_X = np.concatenate((train_X, zero_pad), axis=2)
        zero_pad = np.zeros((test_X.shape[0], test_X.shape[1], 2))
        test_X = np.concatenate((test_X, zero_pad), axis=2)
    elif dataset == 'EthanolConcentration':
        # EthanolConcentration's length: 1751->1755
        zero_pad = np.zeros((train_X.shape[0], train_X.shape[1], 4))
        train_X = np.concatenate((train_X, zero_pad), axis=2)
        zero_pad = np.zeros((test_X.shape[0], test_X.shape[1], 4))
        test_X = np.concatenate((test_X, zero_pad), axis=2)
    elif dataset == 'NATOPS':
        # NATOPS' length: 51->54
        zero_pad = np.zeros((train_X.shape[0], train_X.shape[1], 3))
        train_X = np.concatenate((train_X, zero_pad), axis=2)
        zero_pad = np.zeros((test_X.shape[0], test_X.shape[1], 3))
        test_X = np.concatenate((test_X, zero_pad), axis=2)
    elif dataset == 'PhonemeSpectra':
        # PhonemeSpectra's length: 217->220
        zero_pad = np.zeros((train_X.shape[0], train_X.shape[1], 3))
        train_X = np.concatenate((train_X, zero_pad), axis=2)
        zero_pad = np.zeros((test_X.shape[0], test_X.shape[1], 3))
        test_X = np.concatenate((test_X, zero_pad), axis=2)
    elif dataset == 'SpokenArabicDigits':
        # SpokenArabicDigits's length: 93->96
        zero_pad = np.zeros((train_X.shape[0], train_X.shape[1], 3))
        train_X = np.concatenate((train_X, zero_pad), axis=2)
        zero_pad = np.zeros((test_X.shape[0], test_X.shape[1], 3))
        test_X = np.concatenate((test_X, zero_pad), axis=2)
    elif dataset == 'ERing':
        # ERing's length: 65->66
        zero_pad = np.zeros((train_X.shape[0], train_X.shape[1], 1))
        train_X = np.concatenate((train_X, zero_pad), axis=2)
        zero_pad = np.zeros((test_X.shape[0], test_X.shape[1], 1))
        test_X = np.concatenate((test_X, zero_pad), axis=2)

    dim_num = train_X.shape[1]
    merge_dim = False
    if dim_num > max_dim:
        new_dim_num = int(dim_num / max_dim) + 1 if int(dim_num % max_dim != 0) else int(dim_num / max_dim)
        pad_dim = max_dim - dim_num % max_dim
        zero_pad = np.zeros((train_X.shape[0], pad_dim, train_X.shape[2]))
        train_X = np.concatenate((train_X, zero_pad), axis=1)
        zero_pad = np.zeros((test_X.shape[0], pad_dim, test_X.shape[2]))
        test_X = np.concatenate((test_X, zero_pad), axis=1)
        train_X = rearrange(train_X, 'n (d1 d2) f -> n d1 (d2 f)', d1=max_dim)
        test_X = rearrange(test_X, 'n (d1 d2) f -> n d1 (d2 f)', d1=max_dim)
        merge_dim = True

    le = LabelEncoder()
    le.fit(train_y)
    train_y = le.transform(train_y)
    test_y = le.transform(test_y)

    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2, random_state=20,
                                                      stratify=train_y)

    if not merge_dim:
        train_dataset = data_loader.My_Multivariate_Data(train_X, train_y)
        val_dataset = data_loader.My_Multivariate_Data(val_X, val_y)
        test_dataset = data_loader.My_Multivariate_Data(test_X, test_y)
    else:
        train_dataset = data_loader.My_Multivariate_Data(train_X, train_y, new_dim_num)
        val_dataset = data_loader.My_Multivariate_Data(val_X, val_y, new_dim_num)
        test_dataset = data_loader.My_Multivariate_Data(test_X, test_y, new_dim_num)
    return train_dataset, val_dataset, test_dataset

    return train_dataset, val_dataset, test_dataset

