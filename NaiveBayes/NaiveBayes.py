# 朴素贝叶斯对条件概率做出了条件独立的假设


class NaiveB:
    """
    A realisation of naive bayes
    """
    def __init__(self, classes, feature_values):
        """
        :param classes: the values of classes
        :param feature_values: a list that restore the values that each features can access
        """
        self.classes = classes
        self.pro_prior = [0 for i in range(len(classes))]
        self.pro_con = [[0 for i in range(len(values))] for values in feature_values]

    def train(self, train_data, train_label):
        """
        :param train_data: the datasets of train
        """