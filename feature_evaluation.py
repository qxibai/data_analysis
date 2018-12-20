# 特征评估：威尔克森秩和检验、随机森林特征评估按照基尼指数、Fisher score打分
# 特征选择后进行评价选择的特征子集的好坏

import scipy.stats
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from numpy.linalg import *
from mpl_toolkits.mplot3d import Axes3D


# PCA作图
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
        # 这里的data_trans为每一列为一个样本，想要变成每一行为一个样本，则进行转置换
        data_trans = data_trans.T
        # 注意一定要变成数组形式，不然画图出现ValueError: Masked arrays must be 1-D的错误
        data_trans = data_trans.A
        return eig_value, data_trans


def draw_pca(data_A, label):
    pca1 = Pca(data_A, 3)
    eig_value, data_trans = pca1.data_trans()
    data_trans = np.real(data_trans)
    eig_importance = eig_value / np.sum(eig_value)
    # 做图

    fig = plt.figure()

    ax4 = fig.add_subplot(111, projection='3d')
    ax4.scatter(data_trans[label == -1][:, 0], data_trans[label == -1][:, 1], data_trans[label == -1][:, 2],
                c='green')
    ax4.scatter(data_trans[label == 1][:, 0], data_trans[label == 1][:, 1], data_trans[label == 1][:, 2],
                c="red")
    ax4.set_xlabel("PC1(" + str(round(eig_importance[0] * 100, 2)) + "%)")
    ax4.set_ylabel("PC2(" + str(round(eig_importance[1] * 100, 2)) + "%)")
    ax4.set_zlabel("PC3(" + str(round(eig_importance[2] * 100, 2)) + "%)")
    plt.show()

    plt.figure()
    ax = plt.subplot(111)
    type1 = ax.scatter(data_trans[label == -1][:, 0], data_trans[label == -1][:, 1],
                       40, marker='o', color="green", label='health')
    type2 = ax.scatter(data_trans[label == 1][:, 0],
                       data_trans[label == 1][:, 1], 40, color="red", marker='o', label='ASD')
    plt.legend((type1, type2), ('health', 'ASD'))
    plt.xlabel("PC1(" + str(round(eig_importance[0] * 100, 2)) + "%)")
    plt.ylabel("PC2(" + str(round(eig_importance[1] * 100, 2)) + "%)")
    plt.title("PCA_PRJNA282013")
    plt.show()


def wilcoxon_rank_sum_test(data, label):
    columns = data.shape[1]
    p_value = np.zeros(columns)
    for i in range(columns):
        data_i = data[:, i] # 提取第i列的数据
        data_A = data_i[label == -1]
        data_H = data_i[label == 1]
        res = scipy.stats.mannwhitneyu(data_A, data_H)
        p_value[i] = res[1]
    return p_value


# 选择具有显著差异的物种
def filter(p_value, p, genus_name):
    index = (p_value < p)
    return genus_name[index], index


# 随机森林进行特征评估
def random_forest_feature(data_T, label):
    feat_labels = data_T.columns
    forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
    forest.fit(data_T, label)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(data_T.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
    return feat_labels[indices]


# Fisher score特征打分, label为两类
def fisher_score(data_T, label):
    data_A = np.array(data_T)
    features_number = data_A.shape[1]  # 特征总个数
    class_counts = pd.value_counts(label)
    score = []
    for j in range(features_number):
        data = data_A[:, j]
        all_mean = data.mean()
        between_class = 0
        in_class = 0
        for i in class_counts.index:
            class_data = data[label == i]
            class_mean = class_data.mean()
            between_class += class_counts[i]*(class_mean - all_mean)**2
            in_class += ((class_data - class_mean)**2).sum()
        score.append(between_class/in_class)
    score = pd.Series(score)
    score.index = data_T.columns
    # 降序排序
    score = score.sort_values(ascending=False)
    return score


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
        # 要注意这里得出来的特征向量是一列一列的
        eig_vector = eig(inv_m)[1].T
        eig_vector = eig_vector[eig_index]
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
        # print("线性变换矩阵为："+ str(eig_vector_choice)+'\n')
        # print("线性变换后数据矩阵变为："+str(data_trans)+'\n')
        return eig_value_choice, eig_vector_choice, data_trans

    def draw_picture(self):
        # 这里的做图只根据两类分类做图，多类做图自行修改
        eig_value_choice, eig_vector_choice, data_trans = self.feature_trans()

        plt.figure()
        ax = plt.subplot(111)
        type1 = ax.scatter(data_trans[self.label == -1][:, 0], data_trans[self.label == -1][:, 1],
                            40, marker='o', color="green", label='health')
        type2 = ax.scatter(data_trans[self.label == 1][:, 0],
                    data_trans[self.label == 1][:, 1], 40, color="red", marker='o', label='ASD')
        plt.legend((type1, type2), ('health', 'ASD'))
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("LDA_PRJNA282013")
        plt.show()


def main():
    data = pd.read_csv("/Users/laisenying/PycharmProjects/untitled1/p1/genus_relative_abundance.csv")
    meta = pd.read_csv("/Users/laisenying/PycharmProjects/untitled1/p1/PRJ355023_meta.csv")
    data2 = data.iloc[:, 2:]
    data2.index = data['Genus_absolute_bundance']
    meta.index = meta['run_id']
    # 可能存在空的样本，去掉
    data3 = data2[data2.columns[data2.sum() > 0]]
    # 对数据进行转置换,此时每一行为一个样本
    data_T = data3.T
    # 依据data_T的run_id即index对meta进行排序和过滤
    meta2 = meta.loc[data_T.index, :]
    # 将index修改为样本名
    data_T.index = meta2['sample_name']
    # 过滤掉在所有样本中都不存在的特征
    data_T = data_T.loc[:, ((data_T == 0).sum()) > 0]

    # # 过滤掉在ASD和health类超过百分之60的样本都不存在的特征
    data_ASD = data_T.loc[(meta2[meta2['disease'] == 'ASD'])['sample_name']]
    data_health = data_T.loc[(meta2[meta2['disease'] == 'Health'])['sample_name']]
    ASD_selection = (data_ASD == 0).sum() <= (0.9 * data_ASD.shape[0])
    health_selection = (data_health == 0).sum() <= (0.9 * data_health.shape[0])
    data_T = data_T[data_T.columns[ASD_selection | health_selection]]

    # 对特征进行标准化
    # data_T = data_T/data_T.sum()
    data_A = np.array(data_T)

    label = np.zeros(len(meta2))
    for i in range(len(meta2)):
        if meta2['disease'][i] == 'ASD':
            label[i] = 1
        else:
            label[i] = -1

    # 从而对剩下的data_T进行特征选择
    # -----1、 wilcoxon_rank_sum_test ---- 评估特征重要性
    print("----analysis of wilcoxon_rank_sum_test----")
    p_value = wilcoxon_rank_sum_test(data_A, label)
    genus_name = data_T.columns
    filtered, index = filter(p_value, 0.05, genus_name)
    abundance = {}
    k = 0
    for i in filtered:
        abundance[i] = [data_T[i].sum(), p_value[index][k]]
        k += 1
    abundance = pd.DataFrame(abundance)
    abundance = abundance.T
    abundance.columns = ['number', 'p_value']
    print("ASD与health中具有显著性差异的菌为" + str(filtered))
    print("具有显著性差异的genus数量为:" + str(len(filtered)))
    print(abundance)

    # ----2、 随机森林进行特征重要性评估 ---
    print("---random_forest_feature_evaluation----")
    features = random_forest_feature(data_T, label)

    # ----3、Fisher Score进行特征打分 ----
    print("---fisher score feature evaluation--- ")
    score = fisher_score(data_T, label)

    # --- 4、 根据所选择的特征进行LDA降维作图 ---
    #"""
    # （1）根据显著性差异菌作图
    selection_data = data_T[abundance.index]
    new = FeatureExtraction(np.array(selection_data), label, 2)
    new.draw_picture()
    selection_index = np.trace(new.inv_matrix())
    print("根据显著性差异特征提取后分类性能为：" + str(selection_index))
    # PCA作图
    draw_pca(np.array(selection_data), label)
    #"""

    #"""
    #  (2) 根据随机森林前几名的特征作图
    max_index = -np.inf
    best_i = 1
    best_data = data_T
    classification_index = []
    number_of_features = []
    for i in range(len(features)+1):
        selection_data = data_T[features[:i+1]]
        new = FeatureExtraction(np.array(selection_data), label, 2)
        # 根据tr(Sw-1Sb)判断选择前几个
        selection_index = np.trace(new.inv_matrix())
        classification_index.append(selection_index)
        number_of_features.append(i)
        if selection_index > max_index:
            max_index = selection_index
            best_data = selection_data
            best_i = i+1
    print(classification_index)
    # 绘制随机森林分类性能曲线图
    plt.figure()
    plt.plot(number_of_features[100:120], classification_index[100:120])
    plt.title("random_forest_classification_evaluation")
    plt.show()
    print("最佳选择分类特征为前"+str(best_i)+"个特征")
    print("最佳分类性能指标为："+str(max_index))
    # LDA降维作图
    new = FeatureExtraction(np.array(best_data), label, 2)
    new.draw_picture()
    # PCA作图
    draw_pca(np.array(best_data), label)
    #"""

    #"""
    # 将特征中使得可分性判据下降的特征删除掉
    select_feature = []
    last_index = 0
    for feature in features:
        select_feature.append(feature)
        selection_data = data_T[select_feature]
        new = FeatureExtraction(np.array(selection_data), label, 2)
        # 根据tr(Sw-1Sb)判断选择前几个
        selection_index = np.trace(new.inv_matrix())
        if selection_index > last_index:
            last_index = selection_index
        else:
            select_feature.remove(feature)

    print(select_feature)
    print("特征个数：" + str(len(select_feature)))
    print("性能指标：" + str(last_index))
    select_data = data_T[select_feature]
    new = FeatureExtraction(np.array(select_data), label, 2)
    new.draw_picture()
    # pca作图
    draw_pca(np.array(select_data), label)
    #"""

    #"""
    # (3) 对fisher score进行降低维度作图
    last_index = 0
    selection_feature = []
    for feature in score.index:
        selection_feature.append(feature)
        new = FeatureExtraction(np.array(data_T[selection_feature]), label, 2)
        # 根据tr(Sw-1Sb)判断选择前几个
        selection_index = np.trace(new.inv_matrix())
        if selection_index > last_index:
            last_index = selection_index
        else:
            selection_feature.remove(feature)
    print("最佳特征选择个数为：" + str(len(selection_feature)))
    print("Fisher Score特征提取后分类性能为：" + str(last_index))
    print("所选择的特征为：" + str(selection_feature))
    new = FeatureExtraction(np.array(data_T[selection_feature]), label, 2)
    new.draw_picture()

    # pca作图
    draw_pca(np.array(data_T[selection_feature]), label)
    #"""

    #"""
    # 对fisher score 和 random_forest共同选择的55个重要性特征进行作图
    select_feature = ['Gordonibacter', 'Haemophilus', 'Desulfotomaculum', 'Lactobacillus', 'Selenomonas', 'Clostridioides', 'Eggerthella']
    new = FeatureExtraction(np.array(data_T[select_feature]), label, 2)
    new.draw_picture()
    # pca作图
    draw_pca(np.array(data_T[select_feature]), label)
    #"""

    #"""
    # 对三种特征选择方法共同的16个特征降维显示
    select_feature = ['Nisaea', 'Akkermansia', 'Tyzzerella', 'Abiotrophia', 'Streptomyces', 'Clostridioides', 'Porphyromonas', 'Noviherbaspirillum', 'Staphylococcus', 'marine bacillus NRRLB-14911[genus]', 'Lachnospira', 'Pannonibacter', 'Parvularcula', 'Micromonas', 'Shewanella', 'Garciella']
    new = FeatureExtraction(np.array(data_T[select_feature]), label, 2)
    new.draw_picture()
    # pca作图
    draw_pca(np.array(data_T[select_feature]), label)
    #"""


if __name__ == "__main__":
    main()
