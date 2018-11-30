# !/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from time import clock
from collections import Counter
'''
实现随机森林CART分类树建立，并对特征的重要性进行评估
特征重要性评估基于基尼指数
特征取值为连续类型，输出数据为离散类型
采用后剪枝
'''
"""
树的结点类
"""


class Node:
    def __init__(self, feature=None, data=None, label=None, value=None, left=None,
                 right=None, output=None, loss=0, alpha=0):
        self.feature = feature
        self.data = data
        self.label=label
        self.value = value
        self.left = left
        self.right = right
        self.output = output
        self.loss = loss
        self.alpha = alpha
        '''
            :param feature: 该结点上所选择要分割的特征
            :param value: 该特征的最优切分点s
            :param data: 给结点上的数据集合
            :param left:
            :param right:
            :param output: 该结点单元所判断的输出值，即最优分类结果
            :param loss: 以该结点为子树的根结点时候计算得到的基尼指数
            :param alpha: 与g(t)的值相等
        '''


class CartTree:
    def __init__(self, train_data, train_label, test_data, test_label, thread=0.0):
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        self.thread = thread
        self.alpha = []
        self.feature_importance={}
    '''
        :param train_data: 输入的数据矩阵，每一行为一个样本，列为特征m*d矩阵，即m个样本，d个特征，DataFrame类型
        :param traon_label: 标签, Series类型
        :param thread: 前剪枝，即基尼指数减小程度小于某个值的时候，停止生成树
        :param alpha: 保存每一个结点上的alpha的值, alpha越小表明减去该结点后损失函数减小的程度越小
    '''

    # 计算该数据集中不同label的数量, 并返回各个类的Pi值
    def label_unique_number(self, data, label):
        if len(data) == 0:
            return 0
        label_stats = label.value_counts()
        label_Pi = {}
        for label in label_stats.index:
            label_Pi[label] = label_stats[label]/len(data)
        return label_Pi

    # 计算当前数据的基尼指数
    def cal_gini(self, data, label):
        if len(data) == 0:
            return 0
        label_Pi = self.label_unique_number(data, label)
        gini = 1
        for index in label_Pi.keys():
            gini -= np.power(label_Pi[index], 2)
        return gini

    # 输入结点m上的data, 当前数据中，找最佳特征j，计算最佳切分点s
    def best_js(self, data, label):
        samples = len(data)  #该结点上样本个数
        if samples == 0:
            return 0
        best_g = np.inf
        best_split = 0
        best_feature = data.columns[0]
        # 遍历每一个特征
        for feature in data.columns:
            # 找到每一个特征中的最好的切分点, 计算该特征所能获得的最佳的基尼指数
            best_s = 0
            best_gini = np.inf
            feature_value = data[feature]  # 该特征下的取值
            bins = np.unique(np.sort(feature_value))
            for i in range(len(bins)-1):
                s = (bins[i] + bins[i+1])/2
                data_left, label_left = data[feature_value >= s], label[feature_value >= s]
                n_left = len(data_left)
                data_right, label_right= data[feature_value < s], label[feature_value < s]
                n_right = samples - n_left
                gini_D = (n_left/samples) * self.cal_gini(data_left, label_left) + \
                         (n_right/samples)*self.cal_gini(data_right, label_right)
                if gini_D < best_gini:
                    best_gini = gini_D
                    best_s = s
            if best_gini < best_g:
                best_feature = feature
                best_g = best_gini
                best_split = best_s
        return best_feature, best_split, best_g

    def build_tree(self, data, label):
        if len(data) == 0:
            return Node()
        current_gini = self.cal_gini(data, label)
        # 首先对data选择划分的最优特征，最优切分点，以及划分后计算得到的gini指数
        best_feature, best_split, now_gini = self.best_js(data, label)
        if (current_gini - now_gini) > self.thread:
            print(best_feature)
            self.feature_importance[best_feature] = 0
            data_left, label_left = data[data[best_feature]>=best_split], label[data[best_feature]>=best_split]
            data_right, label_right = data[data[best_feature]<best_split], label[data[best_feature]<best_split]
            current_output = label.value_counts().index[0]
            return Node(feature=best_feature, data=data, label=label, value=best_split,
                        left=self.build_tree(data_left, label_left),
                        right=self.build_tree(data_right, label_right),
                        output=current_output)
        else:
            # 该结点上的数据集已经可以进行预测分类结果，没必要进一步依据特征划分
            # 多类表决法预测输出值预测输出值，是当前单元的预测分类
            current_output = label.value_counts().index[0]
            return Node(data=data, label=label, output=current_output)

    # 计算C(T)
    # node:结点t
    # loss: 以t为根即诶耽的子树Tt的损失函数
    # samples: 结点t包含的样本数量
    # 若之前设置thread为0，未进行前剪枝，那么这里的损失为0
    # 给一个结点，计算当前结点的损失以及包括结点在内的子结点的个数
    def cal_CT(self, node_t):
        samples = len(node_t.data)

        def CT(node, sum):
            if node.feature is not None:
                CT(node.left, samples)
                CT(node.right, samples)
            else:
                n = len(node.data)
                p = n/sum
                node_t.loss += p*self.cal_gini(node.data, node.label)
        CT(node_t, samples)

    # 求以t为根结点时候叶子结点个数
    def number_leave(self, node_t):
        if node_t.feature is None:
            return 1
        else:
            return self.number_leave(node_t.left) + self.number_leave(node_t.right)

    # 计算g(t)，表示剪枝后整体损失函数减小的程度
    def cal_alpha(self, node_t):
        self.cal_CT(node_t)
        CTt = node_t.loss
        T_number = self.number_leave(node_t)
        if T_number == 1:
            node_t.alpha = np.inf
        else:
            Ct = self.cal_gini(node_t.data, node_t.label)
            gt = (Ct - CTt)/(T_number - 1)
            node_t.alpha = gt

    # 输入：单个样本，以及训练的分类树
    # 输出：分类标签
    def sample_predict(self, sample, tree):
        if tree.feature is None:
            return tree.output
        else:
            if sample[tree.feature] >= tree.value:
                return self.sample_predict(sample, tree.left)
            else:
                return self.sample_predict(sample, tree.right)

    # 对某一棵子树用test_data进行验证, 返回错误率
    def test(self, tree):
        samples = len(self.test_data)
        predict_label = np.zeros(samples)
        for i in range(samples):
            predict_label[i] = self.sample_predict(self.test_data.iloc[i], tree)
        # 计算误差
        error = 0
        for i in range(samples):
            if predict_label[i] != self.test_label.iloc[i]:
                error += 1
        print("error_rate:" + str(error/samples))
        return error/samples

    # 递归的求解cart树上每一个结点的alpha的值
    def digui_alpha(self, tree):
        if tree.feature is not None:
            self.cal_alpha(tree)
            self.alpha.append(tree.alpha)
        if tree.left.feature is not None:
            self.digui_alpha(tree.left)
        if tree.right.feature is not None:
            self.digui_alpha(tree.right)

    # 找到当前与输入的alpha值想对应的结点，并将该结点剪掉，生成子树
    # 对cart_tree进行修剪
    def generate_child_tree(self, alpha, b_tree):
        def child_tree(alpha, tree):
            if tree.alpha == alpha:
                tree.feature = None
            else:
                if tree.left.feature is not None:
                    return child_tree(alpha, tree.left)
                if tree.right.feature is not None:
                    return child_tree(alpha, tree.right)
        child_tree(alpha, b_tree)
        return b_tree

    def pruning(self):
        # 首先利用训练集生成cart分类树
        cart_tree = self.build_tree(self.train_data, self.train_label)
        # 从上至下递归地求解cart_tree上每一个结点上的alpha即g(t)的值,此时每一个结点的alpha属性也被修改
        self.digui_alpha(cart_tree)
        # 对alpha值从小到大排序,alpha的第一个值是整棵训练树的根结点，这里我们将其去掉
        self.alpha = self.alpha[1:]
        self.alpha.append(0)  # 加上一个初始值为0，对应于完全未修整的树T0
        self.alpha.sort()
        # 至上而下地进行剪枝，如果node.alpha = self.alpha[i]，则进行剪枝，生成子树Ti
        best_error_rate = self.test(cart_tree)
        best_alpha = 0
        best_tree_index = 0
        for i in range(len(self.alpha)-1):
            T_tree = self.generate_child_tree(self.alpha[i+1], cart_tree)
            error_rate = self.test(T_tree)
            if error_rate < best_error_rate:
                best_error_rate = error_rate
                best_tree_index = i+1
                best_alpha = self.alpha[i+1]
        print("最优的cart树为T" + str(best_tree_index))
        print("该验证数据的错误率为："+str(best_error_rate))
        print("alpha:" + str(best_alpha))
        cart_tree = self.build_tree(self.train_data, self.train_label)
        if best_alpha == 0:
            best_tree = cart_tree
        else:
            self.generate_child_tree(best_alpha, cart_tree)
            best_tree = cart_tree
        return best_tree

    def cal_feature_importance(self, best_tree):
        if best_tree.feature is not None:
            m = len(best_tree.data)
            ml = len(best_tree.left.data)
            mr = len(best_tree.right.data)
            GIm = self.cal_gini(best_tree.data, best_tree.label)
            GIl = self.cal_gini(best_tree.left.data, best_tree.left.label)
            GIr = self.cal_gini(best_tree.right.data, best_tree.right.label)
            print("重要性：" + str(GIm - (ml/m)*GIl - (mr/m)*GIr))
            self.feature_importance[best_tree.feature] += (GIm - (ml/m)*GIl - (mr/m)*GIr)
            self.cal_feature_importance(best_tree.left)
            self.cal_feature_importance(best_tree.right)

    # 外来样本测试入口, data需要为dataframe类型
    def predict(self, data):
        m = len(data)
        predict_labels = np.zeros(m)
        best_tree = self.pruning()
        for i in range(m):
            predict_labels[i] = self.sample_predict(data.iloc[i], best_tree)
        print("样本预测结果：" + str(predict_labels))
        return predict_labels


# 前序遍历所建立的cart结点
def search_cart_tree(cart_tree):
    if cart_tree.feature is None:
        print("输出结果：" + str(cart_tree.output))
        print(cart_tree.label)
    else:
        print("该结点上最优分类特征："+str(cart_tree.feature))
        print("该结点上最优切分点："+str(cart_tree.value))
        print("遍历左子树：")
        search_cart_tree(cart_tree.left)
        print("遍历右子树：")
        search_cart_tree(cart_tree.right)


class RandomForest():
    def __init__(self, data, label, number, features=0.8, frac=0.7, thread=0.0):
        self.data = data
        self.label = label
        self.number = number
        self.features = features
        self.frac = frac
        self.thread = thread
        self.embedding_tree = []
        self.embedding_CartTree = []
        '''
        :param: number: 随机森林中生成的决策树的个数
        :param frac: 随机抽样时候作为训练集合的比例
        :param: features: 每个决策树所选取的特征的个数, 百分数
        :param embedding_tree: 保存每棵决策树生成的分类树
        :param embedding_CartTree 保存每次采样生成的CartTree类对象
        '''

    def sampling(self):
        for i in range(self.number):
            train_data = self.data.sample(frac=1.0, replace=True, random_state=0, axis=0)  #有放回抽样抽取全部
            train_data = self.data.sample(frac=self.frac, replace=False, random_state=0, axis=0)
            train_data = train_data.sample(frac=self.features, replace=False, random_state=0, axis=1)
            train_label = self.label.iloc[train_data.index]
            test_data = self.data.iloc[list(set(self.data.index)-set(train_data.index))]
            test_data = test_data[train_data.columns]
            test_label = self.label.iloc[test_data.index]
            print("正在生成CartTree......")
            build_tree = CartTree(train_data, train_label, test_data, test_label, self.thread)
            best_tree = build_tree.pruning()
            build_tree.cal_feature_importance(best_tree)
            self.embedding_tree.append(best_tree)
            self.embedding_CartTree.append(build_tree)

    # 对单个输入的sample预测结果
    def predict(self, sample):
        output = np.zeros(self.number) # 用来保存每棵树的决策结果
        for i in range(self.number):
            output[i] = self.embedding_CartTree[i].sample_predict(sample, self.embedding_tree[i])
        # 多数表决判断结果
        return pd.Series(output).value_counts().index[0]

    # 对输入的数据集合进行结果判读
    def all_predict(self, samples):
        m = len(samples)
        result = np.zeros(m)
        for i in range(m):
            result[i] = self.predict(samples.iloc[i])
        print("预测结果为：" + str(result))
        return result

    # 对随机森林进行测试，返回错误率
    def test_data_predict(self, test_data, test_label):
        m = len(test_data)
        result = self.all_predict(test_data)
        error = 0
        for i in range(m):
            if result[i] != test_label.iloc[i]:
                error += 1
        print("随机森林测试数据的错误率为：" + str(error/m))
        return error/m

    # 对所有的特征的重要性进行评估
    def feature_importance_evaluation(self):
        importance_list = {}
        for feature in self.data.columns:
            importance_list[feature] = 0
        for i in range(self.number):
            self.embedding_CartTree[i].cal_feature_importance(self.embedding_tree[i])
            for index in self.embedding_CartTree[i].feature_importance.keys():
                importance_list[index] += self.embedding_CartTree[i].feature_importance[index]
        importance_list = Counter(importance_list)  # 从大到小排序
        sums = sum(importance_list.values())
        # 归一化评估重要性
        for index in importance_list.keys():
            importance_list[index] = importance_list[index]/sums
        importance_list = pd.DataFrame(importance_list)
        print("特征重要性程度为：" + str(importance_list))
        return importance_list


def main():
    #测试数据获取 每一行为一个样本 第一列为分类标签
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
    df = pd.read_csv(url, header=None)
    df.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                  'Alcalinity of ash', 'Magnesium', 'Total phenols',
                  'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                  'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
    data = df.iloc[:, 1:]
    label = df.iloc[:, 0]
    '''
    train_df = df.sample(frac=0.7, replace=False, random_state=1, axis=0)
    test_df = df.iloc[list(set(df.index) - set(train_df.index))]
    train_data, train_label = train_df.iloc[:, 1:], train_df.iloc[:, 0]
    test_data, test_label = test_df.iloc[:, 1:], test_df.iloc[:, 0]
    new = CartTree(train_data, train_label, test_data, test_label)
    best_tree = new.pruning()
    new.cal_feature_importance(best_tree)
    print(new.feature_importance)
    search_cart_tree(best_tree)
    '''
    forest = RandomForest(data, label, number=10, features=0.8, frac=0.7, thread=0)
    forest.sampling()
    forest.feature_importance_evaluation()


if __name__ == "__main__":
    t1 = clock()
    main()
    t2 = clock()
    print("time:" + str(t2-t1))
