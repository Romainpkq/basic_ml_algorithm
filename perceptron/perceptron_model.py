# Author: Romain
# Date: 2020-07-10

"""
数据集：Mnist
训练集数量： 60000
测试集数量： 10000
——————————————————————————————
运行结果：
正确率：83.67%
运行时长:25.87s
"""
import time

import pandas as pd
import numpy as np


# 读取数据
def load_data(file_path, n, ):
    """
    从.csv数据集中读取数据
    n: train or test
    """
    file = pd.read_csv(file_path, encoding='utf-8')
    # print(file.head())

    headers = list(file.columns.values)
    label_header = headers[0]


    # 获取所有的标签
    total_labels = file[label_header].drop_duplicates().tolist()
    # print(total_labels)

    # 获取最多的标签的数据
    num_labels = file[[label_header]].groupby(label_header).agg({label_header: 'count'})
    # print(num_labels)

    # K 是最多的标签数
    features = []
    K = num_labels.agg({label_header: "max"}).tolist()[0]

    # 将所有的数据量都统一到最多，差的部分通过在相应的数据中随机选取

    for label in total_labels:
        l = file[file[label_header] == label].values
        num = l.shape[0]

        if n == 'train':
            if num < K:
                num1 = np.random.randint(0, num, K - num)
                l1 = l[num1]
                l = np.vstack((l, l1))
                features.append(l)

            else:
                features.append(l)

        elif n == 'test':
            features.append(l)

    data = np.vstack(features)
    np.random.shuffle(data)
    labels = data[:, 0]

    features = data[:, 1:]/255

    # 处理数据
    # 所有标签小于5的标记为-1， 大于等于5的标记为1
    for i in range(len(labels)):
        if labels[i] >= 5:
            labels[i] = 1
        else:
            labels[i] = -1

    return features, labels


def percetron_train(features, labels):
    # data是label和features的结合
    # 初始化w和b
    print(features.shape)
    print(labels.shape)
    w = np.zeros((1, features.shape[1]))
    b = np.zeros((1, 1))

    iter = 30
    h = 0.0001

    for j in range(iter):
        for i in range(features.shape[0]):
            yi = labels[i].reshape(-1, 1)
            xi = features[i].reshape(features.shape[1], 1)

            if (-1 * yi * (np.dot(w, xi) + b)) >= 0:
                w += h * yi * xi.T
                b += h * yi

        print("iter " + str(j))

    return w, b


def percetron_test(features, labels, w, b):
    num = 0
    for i in range(features.shape[0]):
        xi = features[i].reshape(features.shape[1], 1)
        yi = labels[i].reshape(-1, 1)

        if (-1 * yi * (np.dot(w, xi) + b)) >= 0:
            num += 1

    return 1 - num / labels.shape[0]


if __name__ == '__main__':
    start = time.time()
    path1 = "../MNIST/mnist_train.csv"
    f_train, l_train = load_data(path1, 'train')

    # 获取权重
    w, b = percetron_train(f_train, l_train)

    path2 = "../MNIST/mnist_test.csv"
    f_test, l_test = load_data(path2, 'test')
    precision = percetron_test(f_test, l_test, w, b)

    end = time.time()
    print("precision: %f" % precision)
    print("spend time: %f" % (end - start))

