# 不加任何优化算法以及加速算法的最原始的KNN算法
# 算法如下：
# 计算已知类别数据集中的点与当前点之间的距离；
# 按照距离递增次序排序
# 选取与当前点距离最小的k个点
# 确定前k个点所在类别的出现频率；
# 返回前k个点出现频率最高的类别作为当前点的预测分类

import numpy as np
import pandas as pd


def txt_to_mat(filename):
    fr = open(filename, 'r')
    lines = fr.readlines()
    number_of_lines = len(lines)
    dataMat = np.zeros((number_of_lines, 3))
    classlabel = []
    index = 0

    for line in lines:
        line_to_list = line.strip().split('\t')
        dataMat[index, :] = line_to_list[0:3]
        if line_to_list[3]=='largeDoses':
            line_to_list[3]=3
        elif line_to_list[3]=='smallDoses':
            line_to_list[3]=2
        else:
            line_to_list[3]=1
        classlabel.append(line_to_list[3])
        index += 1
    return dataMat, classlabel


filename = "/Users/laisenying/Documents/数据挖掘源文件/MachineLearningInAction/Ch02/datingTestSet.txt"
dataMat, classlabel = txt_to_mat(filename)


class KNN():
    def __init__(self, k, labels, dataSet):
        # input: newInput: 1 x N
        #        dataset: M x N (M samples, N features)
        #        labels: 1 x M
        #        k : number of neighbors to use for comparison
        # output: the most popular label
        self.k = k
        self.labels = labels
        self.dataSet = dataSet

    def euclidean_distance(self, new_input):
        # 计算训练集样本与新输入样本的所有欧式距离
        return np.sqrt(np.power(self.dataSet - new_input, 2).sum(axis=1))

    def rank_distance(self, distance):
        # 对距离进行从达到小排序，并选出前k个最大的,返回对应的索引数组
        k_index = np.argsort(-distance)[: self.k]
        return k_index

    def decide_labels(self, new_input):
        m_distance = self.euclidean_distance(new_input)
        k_index = self.rank_distance(m_distance)
        # 这里需要首先将labels的list形式转变为np.array形式，不然无法对其进行取索引操作
        label_Ar = np.array(self.labels)
        k_labels = label_Ar[k_index]
        # 利用pd.value_counts进行计数统计
        # 注意这里返回的是一个pd.Series形式，以label作为index，而以出现次数作为data
        counts = pd.value_counts(k_labels)
        counts_max = counts.max()
        max_label = counts[counts == counts_max]
        # max_label也是一个pd.Series形式，它的index是最有可能的分类
        return list(max_label.index)


if __name__ == "__main__":
    test_group = np.array([[1.0, 0.9],[1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
    test_labels = ['A', 'A', 'B', 'B'] # four samples and two classes
    test_X = np.array([1.2, 1.0])
    test = KNN(k=3, labels=test_labels, dataSet=test_group)
    print(str(test_X) + "的最佳分类结果为" + str(test.decide_labels(test_X)))

    test_Y = np.array([0.1, 0.3])
    print(str(test_Y) + "的最佳分类结果为" + str(test.decide_labels(test_Y)))














