"""
main code that you will run
"""

from linear_model import LogisticRegression
from data_handler import load_dataset, split_dataset
from metrics import accuracy
from metrics import precision_score, recall_score, f1_score


if __name__ == '__main__':
    # data load
    X, y = load_dataset()

    # split train and test
    X_train, y_train, X_test, y_test = split_dataset(X, y, 0.2, True)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)


    # training
    # params = dict()
    iteration = 10000
    learning_rate = 0.005
    classifier = LogisticRegression(learning_rate, iteration)
    classifier.fit(X_train, y_train)

    # testing
    y_pred = classifier.predict(X_test)
    # print(y_pred.shape)
    # print(y_test.shape)

    # performance on test set
    print('Accuracy ', accuracy(y_true=y_test, y_pred=y_pred))
    print('Recall score ', recall_score(y_true=y_test, y_pred=y_pred))
    print('Precision score ', precision_score(y_true=y_test, y_pred=y_pred))
    # print('F1 score ', f1_score(y_true=y_test, y_pred=y_pred))
