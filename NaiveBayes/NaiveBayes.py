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
        :param train_data: the datasets of train, an array, (number_data, features)
        :param train_label: the label of according train dataset, an array, (number_data)
        Update the pro_prior and pro_con parameter
        """

        # get the prior probability
        for i in range(len(self.classes)):
            self.pro_prior[i] = train_label.count(self.classes[i])

        for j in train_data:
            pass

