# coding=utf-8
# Author: Romain
# Date: 2020-09-14

'''
数据集：Mnist
训练集数据量：60000
测试集数据量：10000
time:
precision:
'''

from support_function.support import load_data


class Node:
    def __init__(self, val, left=None, right=None):
        """
        :param val: the value of the node, it is data that separate the dataset
        """
        self.val = val
        self.left = left
        self.right = right


class kdTree:
    def __init__(self, features, labels):
        root, left, right = self.sort(features, 0)
        self.root = root
        self.right = right
        self.left = left
        pass

    def sort(self, data, axis):
        data_change = list(enumerate(data))
        data_change = list(sorted(data_change, key=lambda x: x[1][axis], reverse=False))
        root_index = data_change[len(data_change)//2][0]
        root = data[root_index]
        left_index = [data_change[i][0] for i in range(len(data_change)//2)]
        right_index = [data_change[i][0] for i in range(len(data_change)//2)]
        left = data[left_index]
        right = data[right_index]

        return root, left, right
