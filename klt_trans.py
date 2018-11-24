import numpy as np
import pandas as pd
from numpy.linalg import *
import matplotlib.pyplot as plt


# 从类均值中提取判别信息,K-L变换
class klt_mean():
    def __init__(self, data, label, d):
        '''

        :param data: 输入的数据矩阵，m*n array，每一行为一个样本
        :param label: 类别标签，m*1 array
        :param d: 想要提取的特征维度
        '''
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
            mean_all += Pi[index] * mean_i[index]
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

    # 计算类内离散度矩阵Sw，Sw作为K-L变换的产生矩阵
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

    # 用Sw作为产生矩阵进行K-L变换，求解本征值和本征向量
    def eig_Sw(self):
        Sw = self.in_matrix()
        eig_value = eig(Sw)[0]
        eig_vector = eig(Sw)[1].T
        return eig_value, eig_vector

    # 计算新特征的分类性能指标
    def J_choice(self):
        Sb = self.between_matrix()
        eig_value, eig_vector = self.eig_Sw()
        # 用于存储分类性能指标
        J_decision = np.zeros(len(Sb))
        for i in range(len(Sb)):
            a = np.mat(eig_vector[i]) * np.mat(Sb)* np.mat(eig_vector[i]).T
            b = eig_value[i]
            J_decision[i] = (a/b)[0,0]
        return J_decision

    # 对性能指标排序，选择排名前d的特征向量组成线性变换特征，并返回新的数据矩阵
    def data_transformation(self):
        J_decision = self.J_choice()
        eig_value, eig_vector = self.eig_Sw()
        index = np.argsort(-J_decision)
        eig_vector_choice = eig_vector[index][:self.d]
        data_trans = np.mat(eig_vector_choice) * self.data.T
        data_trans = np.array(data_trans.T)
        return data_trans

# 类中心化特征向量中分类信息的提取，相当于基于类方差来提取信息
class klt_var():
    def __init__(self, data, label, d):
        '''

        :param data: 输入的数据矩阵，m*n array，每一行为一个样本
        :param label: 类别标签，m*1 array
        :param d: 想要提取的特征维度
        '''
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
            mean_all += Pi[index] * mean_i[index]
        # mean_i为字典类型，mean_all为array类型
        return mean_i, mean_all

    # 计算类内离散度矩阵Sw，Sw作为K-L变换的产生矩阵
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

    # 计算类内离散度矩阵Sw的特征值和特征向量
    def eig_Sw(self):
        Sw = self.in_matrix()
        eig_value = eig(Sw)[0]
        eig_vector = eig(Sw)[1].T
        return eig_value, eig_vector

    # 评估特征分类性能，这里以总体所熵来评估
    def data_transformation(self):
        eig_value, eig_vector = self.eig_Sw()
        print(eig_value[1])
        # data_trans为每一列为一个样本,每一行为一个特征
        data_trans = np.mat(eig_vector) * np.mat(self.data).T
        Pi = self.LabelPossibility()
        # 用于存储每一个判据的分类性能
        J_choice = np.zeros(len(data_trans))
        # 遍历每一个特征
        for j in range(len(data_trans)):
            r_ij = 0
            # 遍历每一个分类
            for i in Pi.keys():
                data_j = data_trans[j]
                data_ij = data_j[:, self.label == i]  # 第i类的第j个特征的数据集
                a = rij = Pi[i]*np.var(data_ij, ddof=1)
                b = eig_value[j]
                rij = a/b
                r_ij -= rij*np.log10(rij)
            J_choice[j] = r_ij
        index = np.argsort(J_choice)
        eig_vector_choice = eig_vector[index][:self.d]
        data_trans = np.mat(eig_vector_choice)*np.mat(self.data).T
        data_trans = data_trans.T
        data_trans = data_trans.A
        return data_trans


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

    kl1 = klt_var(data_A, label, 2)
    data_trans = kl1.data_transformation()
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
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()
    kl1 = klt_mean(data_A, label, 2)
    data_trans = kl1.data_transformation()
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
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()
