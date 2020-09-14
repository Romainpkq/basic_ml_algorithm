import numpy as np
import pandas as pd


def load_data(file_path, n):
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