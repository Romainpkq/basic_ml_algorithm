# coding=utf-8
# Author: Romain
# Date: 2020-09-08

'''
数据集：Mnist
训练集数据量：60000
测试集数据量：10000
time: 34.017232
precision: 0.810381
'''

import numpy as np
import time
from support_function.support import load_data


# def loadData(file):
#     '''
#     加载Mnist数据集
#     :param file: 数据集路径
#     :return: list形式的数据集以及标签
#     如果采用这汇总方式加载数据，存在一些问题
#     '''
#     datasets = pd.read_csv(file)
#     column_names = list(datasets.columns.values)[1:]
#     # change the label, if label >=5, label=1, else label=-1
#
#     def transform_label(x):
#         if x >= 5:
#             return 1
#         else:
#             return -1
#
#     datasets['5'] = datasets['5'].apply(lambda x: transform_label(x))
#     labels = list(datasets.iloc[:, 0])
#
#     # get the features of all the data
#     features = []
#
#     for index, row in datasets.iterrows():
#         # normalization
#         row_data = [int(row[name])/255 for name in column_names]
#         features.append(row_data)
#
#     return features, labels


class Perceptron:
    """
    This class is to implement a percetron.
    """
    def __init__(self, num_features, set_bias=True):
        self.num_features = num_features
        self.w = np.random.uniform(-1, 1, size=(num_features, 1))

        if set_bias:
            self.bias = np.random.uniform(-1, 1)
        else:
            self.bias = 0

    def loss_function(self, inputs, labels):
        """
        param inputs: ndarray:
            size: (num_example, num_features)
        param labels: ndarray:
            size: (num_example, 1)
        return loss: the loss of all the data
        """
        result = np.dot(inputs, self.w) + self.bias   # (num_example, 1)
        result = - result * labels   # (num_example)
        #print("result:", result)
        loss = 0
        wrong_answer = 0
        for num in result:
            if num > 0:
                loss += num
                wrong_answer += 1

        return loss/len(result), (1 - wrong_answer/result.shape[0])

    def train(self, inputs, labels, num_epochs, learning_rate=1e-4):
        """
        param inputs: ndarray:
            size: (num_example, num_features)
        param labels: ndarray:
            size: (num_example, 1)
        param num_steps:
            the loop steps
        """
        num_examples = inputs.shape[0]

        for i in range(num_epochs):
            for j in range(num_examples):
                # index = np.random.randint(0, num_examples)
                result = np.dot(inputs[j], self.w) + self.bias

                if result * labels[j] <= 0:
                    inputs1 = np.reshape(inputs[j], (self.num_features, 1))
                    self.w += learning_rate * labels[j] * inputs1
                    self.bias += learning_rate * labels[j]

            loss1, precision = self.loss_function(inputs, labels)
            print("After %d epochs, the loss is %f, the train precision is %f" % (i, loss1, precision))

            print("The training step ends.")

    def test(self, inputs, labels):
        """
        param inputs: ndarray:
            size: (num_example, num_features)
        param labels: ndarray:
            size: (num_example, 1)
        param num_steps:
            the loop steps
        """
        result = np.dot(inputs, self.w) + self.bias
        result = result * labels
        # print('result.shape:', result.shape)
        result1 = (result > 0)
        return np.sum(result1 == True)/result.shape[0]


if __name__ == '__main__':
    print("start")
    start = time.time()
    train_path = '../MNIST/mnist_train.csv'
    test_path = '../MNIST/mnist_test.csv'
    # (train_features, train_labels) = loadData(train_path)
    # (test_features, test_labels) = loadData(test_path)
    (train_features, train_labels) = load_data(train_path, 'train')
    (test_features, test_labels) = load_data(test_path, 'test')

    train_features = np.asarray(train_features)
    train_labels = np.asarray(train_labels)
    train_labels = np.reshape(train_labels, (train_labels.shape[0], 1))

    test_features = np.asarray(test_features)
    test_labels = np.asarray(test_labels)
    test_labels = np.reshape(test_labels, (test_labels.shape[0], 1))

    num_features = train_features.shape[1]

    # construct model
    p = Perceptron(num_features)
    p.train(train_features, train_labels, 50)
    precision = p.test(test_features, test_labels)
    end = time.time()
    print("precision: %f" % precision)
    print("spend time: %f" % (end - start))


