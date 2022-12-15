import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression:
    def __init__(self, learning_rate, iteration):
        """
        figure out necessary params to take as input
        :param params:
        """
        # todo: implement

        self.learning_rate = learning_rate
        self.iteration = iteration
        self.weight = 0
        self.cons = 0

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        # assert X.shape[0] == y.shape[0]
        # assert len(X.shape) == 2
        # todo: implement

        evidence_size = X.shape[0]
        observation_size = X.shape[1]
        self.weight = np.zeros((evidence_size, 1))

        # print(X.shape)
        # print(self.weight.shape)
        # print(X.T.shape)
        # print(y.shape)

        for i in range(self.iteration):
            y_pred = np.dot(self.weight.T, X) + self.cons
            # print(y_pred.shape)
            p = sigmoid(y_pred)
            # print(y.shape)
            # print(p.shape)

            cost = -(1 / observation_size) * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

            # gradient descent
            dWeight = (1 / observation_size) * np.dot((p - y), X.T)
            dCons = (1 / observation_size) * np.sum(p - y)

            self.weight = self.weight - self.learning_rate * dWeight.T
            self.cons = self.cons - self.learning_rate * dCons

            # if i%(self.iteration / 10) == 0:
            #     print('cost after iteration ', i, ':', cost)

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X:
        :return:
        """
        # todo: implement

        y_pred = np.dot(self.weight.T, X) + self.cons
        p = sigmoid(y_pred)
        p = p >= 0.5
        p = np.array(p, dtype=int)
        # shape of 'p' is (1 x test_size)

        return p

