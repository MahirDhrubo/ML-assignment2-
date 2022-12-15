import numpy as np
import pandas as pd
import random


def load_dataset():
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :return:
    """
    # todo: implement

    file_name = 'data_banknote_authentication.csv'
    data = pd.read_csv(file_name)
    X = data.iloc[:, 0:4].values
    X = X.T
    y = data.iloc[:, 4].values
    y = y.reshape(1, y.shape[0])

    return X, y


def split_dataset(X, y, test_size, shuffle):
    """
    function for spliting dataset into train and test
    :param X:
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """
    # todo: implement.
    dataSize = y.shape[1]
    test_size = int(test_size * dataSize)
    arr = np.arange(0, dataSize, dtype=int)
    if shuffle:
        np.random.shuffle(arr)
    X_test = X[:, arr[0:test_size]]
    y_test = y[:, arr[0:test_size]]
    X_train = X[:, arr[test_size:dataSize]]
    y_train = y[:, arr[test_size:dataSize]]

    # X_train, y_train, X_test, y_test = None,None,None,None
    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y):
    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X:
    :param y:
    :return:
    """
    # todo: implement
    sample_size = X.shape[1]
    X_sample = np.empty(X.shape, dtype=int)
    y_sample = np.empty(y.shape, dtype=int)
    
    for i in range(sample_size):
        idx = random.randint(0, sample_size-1)
        X_sample[:, i] = X[:, idx]
        y_sample[:, i] = y[:, idx]

    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape
    return X_sample, y_sample
