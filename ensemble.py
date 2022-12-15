from data_handler import bagging_sampler
import copy
import numpy as np
import random


class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator):
        """
        :param base_estimator:
        :param n_estimator:
        :return:
        """
        # todo: implement
        self.base_estimator = base_estimator
        self.n_estimator = n_estimator

        self.model = []
        for i in range(n_estimator):
            self.model.append(copy.deepcopy(base_estimator))

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        # assert X.shape[0] == y.shape[0]
        # assert len(X.shape) == 2
        # todo: implement
        size = X.shape[1]
        arr = np.arange(size)
        for i in range(self.n_estimator):
            X_sample, y_sample = bagging_sampler(X, y)
            print('estimator : ', i)
            # print(X_smaple.shape)
            # print(y_smaple.shape)
            self.model[i].fit(X_sample, y_sample)



    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        apply majority voting
        :param X:
        :return:
        """
        # todo: implement
        test_size = X.shape[1]
        y_pred = np.zeros((self.n_estimator, test_size), dtype=int)
        for i in range(self.n_estimator):
            y_pred[i] = self.model[i].predict(X)
        
        y = np.zeros((1, test_size), dtype=int)
        for i in range(test_size):
            y[0][i] = np.bincount(y_pred[:, i]).argmax()
        
        # print(y.shape)

        return y
