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
# 过程
# 1. 获取所有的数据点{(X11, X12, ..., X1n), ...}, 初始化 left=[], right = []
# 2. for i in range(n):
#    sort left, sort right
#    median_left = value1
#    median_right = value2
#
# def tree(left):
#   return Node(value,)


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
        self.root = self.construct(features, labels, 0)

    def construct(self, ls, labels, axis):
        """
        :param ls: the list of all the data.
        :param axis: according axis dimension to split the data
        return:
            A Node(val, None, None)
        """
        if len(ls) == 1:
            return Node(ls[0])

        else:
            ls1 = ls[ls[:, axis].argsort()]
            labels1 = labels[ls[:, axis].argsort()]
            node = Node(ls[len(ls)//2], None, None)
            node.right = self.construct(ls1[len(ls)//2:], labels1[len(ls)//2:], axis + 1)
            node.left = self.construct(ls1[:len(ls)//2], labels1[:len(ls)//2], axis+1)
            return node

    def search(self, target):
        """
        :param target: the target point x
        """


