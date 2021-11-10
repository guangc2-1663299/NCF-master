import numpy as np 
import pandas as pd 
import scipy.sparse as sp

import torch.utils.data as data

import config


def load_all(test_num=100):
    impression_train = pd.read_csv('/Users/guangyuan/Desktop/capstone/ncf jupyter文件/train_impression.csv')
    impression_train = impression_train.drop('Unnamed: 0', axis=1)
    #impression_train
    impression_test_index = pd.read_csv('/Users/guangyuan/Desktop/capstone/ncf jupyter文件/test_data_index.csv')
    impression_test_index = impression_test_index[['user_index', 'mlog_index']]
    impression_train = impression_train[impression_train['isClick'] == 1]
    impression_train_index = impression_train[['user_index', 'mlog_index']]
    ##build a sparse matrix for train data
    train_data = impression_train_index.values.tolist()
    user_num = impression_train_index['user_index'].max() + 1
    item_num = impression_train_index['mlog_index'].max() + 1
    # length of df = 2870438
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    test_data = impression_test_index
    train_data = impression_train_index
    return train_data, test_data, user_num, item_num, train_mat


class NCFData(data.Dataset):
    def __init__(self, features,
                num_item, train_mat=None, num_ng=0, is_training=None):
        super(NCFData, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
        """
        self.features_ps = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training
        self.labels = [0 for _ in range(len(features))]

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'

        self.features_ng = []
        for x in self.features_ps:
            u = x[0]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_ng.append([u, j])

        labels_ps = [1 for _ in range(len(self.features_ps))]
        labels_ng = [0 for _ in range(len(self.features_ng))]

        self.features_fill = self.features_ps + self.features_ng
        self.labels_fill = labels_ps + labels_ng

    def __len__(self):
        return (self.num_ng + 1) * len(self.labels)

    def __getitem__(self, idx):
        features = self.features_fill if self.is_training \
                    else self.features_ps
        labels = self.labels_fill if self.is_training \
                    else self.labels

        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]
        return user, item ,label
