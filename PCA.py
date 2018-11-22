# 主成分分析

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import *


class Pca():
    def __init__(self, data, d):
        '''

        :param data: 样本数据矩阵，这里为m*n，每一行为一个样本，即m个样本，n个特征
        :param d: 想要提取的特征数量
        '''
        self.data = data
        self.d = d

    # 对样本矩阵进行零均值化, 并且还要对样本进行转置，使得每一列为一个样本
    def zero_mat(self):
        return (self.data - self.data.mean(axis=0)).T

    # 求协方差矩阵
    def cov_mat(self):
        data_mat = self.zero_mat()
        data_cov = np.mat(data_mat)*np.mat(data_mat).T/len(self.data)
        return data_cov

    # 计算协方差矩阵的特征值和特征向量，并让特征向量按照特征值大到小排序
    def eig_data(self):
        data_cov = self.cov_mat()
        eig_index = np.argsort(-eig(data_cov)[0])
        eig_value = eig(data_cov)[0][eig_index]
        # 要注意这里的特征向量是一列一列，所以需要先进行转置换
        eig_vector = eig(data_cov)[1].T
        eig_vector = eig_vector[eig_index]
        return eig_value, eig_vector

    def data_trans(self):
        zero_data = self.zero_mat()
        eig_value, eig_vector = self.eig_data()
        data_trans = eig_vector[:self.d]*zero_data
        eig_value_choice = eig_value[:self.d]
        print("特征变换矩阵为："+str(eig_vector[:self.d]))
        print("前d个特征值为："+ str(eig_value_choice))
        # 这里的data_trans为每一列为一个样本，想要变成每一行为一个样本，则进行转置换
        data_trans = data_trans.T
        # 注意一定要变成数组形式，不然画图出现ValueError: Masked arrays must be 1-D的错误
        data_trans = data_trans.A
        return eig_value, data_trans


if __name__ == "__main__":
    # 数据读取与处理，对数据进行分类读取，并转换为数据矩阵，每一行为一个特征向量
    data = pd.read_csv("genus_relative_abundance.csv")
    meta = pd.read_csv("PRJ355023_meta.csv")
    # 删除掉数据的第一列
    data2 = data.iloc[:, 2:]
    # 将属的名字作为每一行的index
    data2.index = data['Genus_absolute_bundance']
    # 将run_id作为meta文件的index
    meta.index = meta['run_id']
    # 去掉空的序列
    data3 = data2[data2.columns[data2.sum() > 0]]
    # 对数据进行转置换，此时变为每一行为一个样本
    data_T = data3.T
    # 依据run_id对meta进行过滤和重排, 要注意loc是针对label进行定位的,而iloc则是针对多少行多少列定位的
    meta2 = meta.loc[data_T.index, :]
    # 将data3的run_id index修改为样本名
    data_T.index = meta2['sample_name']
    # 对数据特征进行筛选，过滤掉小于0.001的特征
    data_T = data_T[data_T.columns[data_T.sum() > 0]]
    data_T = data_T[data_T.columns[data_T.sum() < 1]]
    print("过滤后特征数量为：" + str(len(data_T.columns)))
    # 提取数据矩阵，每一行构成一个特征向量（一个样本）
    data_A = np.array(data_T)
    # 样本类别储存
    label = np.zeros(len(data_A))
    for i in range(len(data_A)):
        if meta2['disease'][i] == 'ASD':
            label[i] = 1
        else:
            label[i] = 0
    cars_label = []
    # cars表列表储存
    for i in range(len(label)):
        if 35 <= meta2['CARS'][i] < 40 :
            cars_label.append(2)
        elif 40 <= meta2['CARS'][i] < 45 :
            cars_label.append(3)
        elif  meta2['CARS'][i] >= 45 :
            cars_label.append(4)
        else:
            cars_label.append(1)
    cars_label = np.array(cars_label)
    pca1 = Pca(data_A, 2)
    eig_value, data_trans = pca1.data_trans()
    eig_importance = eig_value/np.sum(eig_value)
    print(data_trans[cars_label == 2][:, 0])
    # 做图
    plt.figure()
    axes = plt.subplot(111)
    type1 = axes.scatter(data_trans[label == 0][:, 0], data_trans[label == 0][:, 1], color="green", label='Health')
    type2 = axes.scatter(data_trans[cars_label == 2][:, 0],
                       data_trans[cars_label == 2][:, 1], 30, color="red", marker='o', label='low')
    type3 = axes.scatter(data_trans[cars_label == 3][:, 0],
                       data_trans[cars_label == 3][:, 1], 40, color="red", marker='<', label='moderate')
    type4 = axes.scatter(data_trans[cars_label == 4][:, 0],
                       data_trans[cars_label == 4][:, 1], 50, color="red", marker='s', label='severe')
    plt.legend((type1, type2, type3, type4), ('Health', 'low', 'moderate', 'severe'))
    plt.xlabel("PC1:"+ str(round(eig_importance[0]*100, 2)) + "%")
    plt.ylabel("PC2" + str(round(eig_importance[1]*100, 2)) + "%")
    plt.show()
