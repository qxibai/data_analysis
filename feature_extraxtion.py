# 基于类间的可分性对特征进行线性变换降维
# 画图的相关地方需要自行调整

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import *


class FeatureExtraction():
    '''
            基于类间的可分性对特征进行线性变换和提取
            :param self:
            :param data: 输入的数据矩阵，这里每一行为一个特征向量（一个样本），m*n
            :param label: 标签，为m*1的array类型
            :param d: 想要提取的特征数量
            :return:
    '''
    def __init__(self, data, label, d):
        self.data = data
        self.label = label
        self.d = d

    # 计算Pi,Pj, 保存类别的概率
    def LabelPossibility(self):
        m = len(self.label)
        label_stats = pd.value_counts(pd.Series(self.label))
        Pi = {}
        for i in range(len(label_stats)):
            # 要注意这里的索引需要用index的名字, index为类名
            index = label_stats.index[i]
            Pi[index] = label_stats[index]/m
        return Pi

    # 计算每一类的均值向量以及总的均值向量
    def Class_mean(self):
        Pi = self.LabelPossibility()
        mean_i = {}
        mean_all = np.zeros(self.data.shape[1])
        for index in Pi.keys():
            # 注意这里的label需要为array类型
            mean_i[index] = self.data[self.label == index].mean(axis=0)
            mean_all += Pi[index]*mean_i[index]
        # mean_i为字典类型，mean_all为array类型
        return mean_i, mean_all

    # 计算类间离散度矩阵
    def between_matrix(self):
        Pi = self.LabelPossibility()
        mean_i, mean_all = self.Class_mean()
        Sb = np.zeros([self.data.shape[1], self.data.shape[1]])
        for i in Pi.keys():
            Sb += Pi[i]*np.mat(mean_i[i]-mean_all).T * np.mat(mean_i[i]-mean_all)
        return Sb

    # 计算类内离散度矩阵
    def in_matrix(self):
        mean_i, mean_all = self.Class_mean()
        Pi = self.LabelPossibility()
        # 计算每一类的(xk-mi)(xk-mi)^T之和
        # class_number: 类别数量
        Sw = np.zeros([self.data.shape[1], self.data.shape[1]])
        for index in Pi.keys():
            data_i = self.data[self.label == index]
            Sw_index = np.zeros([self.data.shape[1], self.data.shape[1]])
            for i in range(len(data_i)):
                Sw_index += np.mat(data_i[i]-mean_i[index]).T * np.mat(data_i[i]-mean_i[index])
            Sw_index = Pi[index] * Sw_index/(len(data_i)-1)
            Sw += Sw_index
        return Sw

    # 计算判Sw^(-1)*Sb, Sw与Sb都是对称阵
    def inv_matrix(self):
        Sw = self.in_matrix()
        Sb = self.between_matrix()
        try:
            inv_Sw = np.mat(inv(Sw))
        except LinAlgError:
            print("类内离散度矩阵Sw不可逆,可以对特征进行过滤一下")
        else:
            return np.mat(inv_Sw)*np.mat(Sb)

    # 计算矩阵Sw^(-1)*Sb的特征值
    def eig_matrix(self):
        inv_m = self.inv_matrix()
        # 调用eig()计算特征值和特征向量
        # 对特征值排序, 从大到小
        eig_index = np.argsort(-eig(inv_m)[0])
        eig_vector = eig(inv_m)[1][eig_index]
        eig_value = eig(inv_m)[0][eig_index]
        return eig_value, eig_vector

    def feature_trans(self):
        eig_value, eig_vector = self.eig_matrix()
        # 现在提取出来的特征向量是d*n的矩阵(d<n)，原样本数据是m*n的矩阵,因此我们需要对样本进行转置变为n*m的矩阵，即每一列为一个样本
        eig_vector_choice = eig_vector[:self.d]
        eig_value_choice = eig_value[:self.d]
        # 对样本数据进行转置，使的每一类为一个特征向量
        # 这里的data_trans为每一列为一个d维向量构成一个样本，想要以每一行作为一个样本时，需要进行转置换,并换回数组模式
        # 因为在后面画图的时候需要用到Array形式的数据
        data_trans = np.mat(eig_vector_choice) * np.mat(self.data.T)
        data_trans = data_trans.T
        data_trans = data_trans.A
        print("线性变换矩阵为："+ str(eig_vector_choice)+'\n')
        print("线性变换后数据矩阵变为："+str(data_trans)+'\n')
        return eig_value_choice, eig_vector_choice, data_trans

    def draw_picture(self, cars_label):
        # 这里的做图只根据两类分类做图，多类做图自行修改
        eig_value_choice, eig_vector_choice, data_trans = self.feature_trans()
        plt.figure()
        ax = plt.subplot(111)
        colorlabel = []
        label_index = pd.value_counts(pd.Series(label)).index
        for i in range(len(self.label)):
            if label[i] == 0:
                colorlabel.append('green')
            else:
                colorlabel.append('red')
        type1 = ax.scatter(data_trans[self.label == 0][:, 0], data_trans[self.label == 0][:, 1],20, color="green", label='Health')
        type2 = ax.scatter(data_trans[cars_label == 2][:, 0],
                    data_trans[cars_label == 2][:, 1], 30, color="red", marker='o', label='low')
        type3 = ax.scatter(data_trans[cars_label == 3][:, 0],
                    data_trans[cars_label == 3][:, 1], 40, color="red", marker='<', label='moderate')
        type4 = ax.scatter(data_trans[cars_label == 4][:, 0],
                    data_trans[cars_label == 4][:, 1], 50, color="red", marker='s', label='severe')
        plt.legend((type1, type2, type3, type4), ('Health', 'low', 'moderate', 'severe'))
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.show()


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
    data_T = data_T[data_T.columns[data_T.sum() > 0.1]]
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
    print(meta2['CARS'])
    cars_label = []
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
    # 进行特征提取
    new = FeatureExtraction(data_A, label, 2)
    eig_value_choice, eig_vector_choice, data_trans = new.feature_trans()
    new.draw_picture(cars_label)
