# Kmeans算法思想如下
# 选取数据空间中的K个对象作为初始中心，每个对象代表一个聚类中心；
# 对于样本中的数据对象，根据它们与这些聚类中心的欧式距离，按距离最近的准则将它们分到距离它们最近的聚类中心。
# 对于所有已经被标记（分类）的点，重新计算类均值（例如所有标记为i的样本的每个特征的均值，将所计算得到的均值作为新的类中心），并计算目标函数的值。
# 判断聚类中心和目标函数的值是否发生改变，若不变，则输出结果，若改变，则返回2）


# /Users/laisenying/Documents/数据挖掘源文件/MachineLearningInAction/Ch10/testSet.txt
import numpy as np
import matplotlib.pyplot as plt


# --------------kmeans算法实现----------

# 计算一个样本与另外多个个样本的欧式距离,并返回距离构成的列表
def euclidean_distance(dataset, one_sample):
    x = np.tile(one_sample, [len(dataset), 1])  # 将传入的样本n维向量扩展成与dataMat同维度的矩阵
    return np.sqrt(np.power(x - dataset, 2).sum(axis=1))  # 返回输入的点与训练集中所有点的距离构成的数组


class Kmeans():
    def __init__(self, k, dataMat, max_iterations, diff):
        self.k = k
        self.dataMat = dataMat
        self.max_iterations = max_iterations
        self.diff = diff

    # 计算所有样本与已经设置为初始化的样本点的欧氏距离之和，设已经初始化了的样本点存放在Kcent矩阵中
    def init_distance(self, Kcent):
        rows = self.dataMat.shape[0]
        n = Kcent.shape[0]  # 已经初始化了的样本点的个数
        init_dist = np.zeros((rows, n))  # 建立空矩阵，每一列用来储存训练集中所有点与第i个初始化点的距离
        # 需要计算每一个样本点与Kcent中第i个初始化点的距离列表
        for i in range(n):
            init_dist[:, i] = euclidean_distance(self.dataMat, Kcent[i, :])
        return init_dist.sum(axis=1)  # 返回训练集中所有点与已经初始化的点的距离之和构成的数组

    # 随机选取一点作为初始质心,返回选择的初始质心的坐标
    def rand_oneCent(self):
        rows = self.dataMat.shape[0] # 读取矩阵的行数，即点的数量
        n = np.random.randint(rows) # 随机选取一行
        return dataMat[n, :]

    # 根据获得的第一个初始坐标，选取初始化的k个质心
    # rand_onecent为初始化的第一个质心的坐标
    def rand_kCent(self, rand_onecent):
        rows, columns = dataMat.shape  # 获得训练集矩阵的行和列数
        Kcent = np.zeros((self.k, columns))  # 建立K*d维矩阵
        Kcent[0, :] = rand_onecent  # 获得初始质心坐标,并存储在Kcent中
        for i in range(self.k - 1):
            dist_list = self.init_distance(Kcent[0:i+1, :])  # 获得所有样本与质心初始化质心的距离构成的数组
            index = dist_list.argmax()  # 获得与与初始质心距离最大的样本点的索引
            Kcent[i+1, :] = self.dataMat[index, :]  # 将距离初始质心最大的那个点储存起来
        return Kcent

    # 现在需要计算训练集合中每一个样本点与到K个质心的欧氏距离，并将该样本点归为最近的那个点
    # kCent为k个质心的坐标构成的矩阵
    def create_clusters(self, kCent):
        rows,columns = self.dataMat.shape
        # 建立一个储存数据类标签的列表
        label = []
        # 计算一个样本与K个质心的距离
        for i in range(rows):
            # 获得欧氏距离最小的标签，即类编号，即dataMat中第i个数据属于第index类
            index = euclidean_distance(self.dataMat[i, :], kCent).argmin()
            label.append(index)
        return label

    # 更新质心位置，需要重新计算所属于每一个类的样本点的平均值,并将距离平均值最近的那个点作为新的质心
    # label为先前已经归了类的数据的类标签列表，list类型
    def update_cent(self, label):
        # 需要先将label转换为数组类型，为了后续label == i判断时候获得索引列表
        label_ar = np.array(label)
        # 建立用于存储新的质心位置的矩阵
        new_kcent = np.zeros((self.k, self.dataMat.shape[1]))
        for i in range(self.k):
            select = (label_ar == i)  # 获得同属于第i类的样本点的索引
            data = self.dataMat[select] # 筛选出同属于第i类的数据
            # 计算同属于第i类样本点坐标的中心，即平均值
            mean_cent = data.mean(axis = 0)
            # 找到该类到样本坐标中心的最近的坐标
            index = euclidean_distance(data, mean_cent).argmin()
            new_kcent[i, :] = data[index, :]  # 将更新后的坐标位置存储下来
        return new_kcent

    # 判断是否停止聚类,即质心的位置是否不怎么变化了
    # 相当于kmeans类的核心方法
    def predict(self):
        # 首先初始化第一个质心坐标
        rand_onecent = self.rand_oneCent()
        # 再初始化获得K个质心坐标
        Kcent = self.rand_kCent(rand_onecent)
        # 进行迭代，直到算法收敛,或者超过最大迭代次数
        for i in range(self.max_iterations):
            # 对训练集合的数据归类
            label = self.create_clusters(Kcent)
            old_Kcent = Kcent  # 保存旧的质心坐标
            # 对质心坐标更新位置
            Kcent = self.update_cent(label)
            # 判断新的质心位置是否与之前的质心位置距离小于self.diff
            # 即两个每个对应质心在每一个维度上的差值都不能小于self.diff
            if abs(Kcent-old_Kcent).max() < self.diff:
                time = i # 返回最终迭代的次数
                break
        # 最终的数据标签为,并数组化
        label = np.array(self.create_clusters(Kcent))
        # 返回最终归类的类标签列表，以及质心坐标
        return label, Kcent, time

if __name__ == "__main__":
    def txt_to_mat(filename):
        fr = open(filename, 'r')
        lines = fr.readlines()
        dataMat = np.zeros((len(lines), 2))
        index = 0
        for line in lines:
            line_to_list = line.strip().split('\t')
            dataMat[index, :] = line_to_list[0:2]
            index += 1
        return dataMat

    filename = '/Users/laisenying/Documents/数据挖掘源文件/MachineLearningInAction/Ch10/testSet.txt'
    dataMat = txt_to_mat(filename)

    print(dataMat)
    plt.figure()
    plt.scatter(dataMat[:, 0], dataMat[:, 1])
    plt.show()

    # kmeans算法执行
    test1 = Kmeans(4, dataMat, 1000000000, 0.0000000001)
    label, Kcent, time = test1.predict()
    #绘图
    plt.figure()
    for i in range(len(label)):
        plt.scatter(dataMat[label == i][:, 0], dataMat[label == i][:, 1])
    plt.scatter(Kcent[:, 0], Kcent[:, 1],color="black", s=35)
    plt.show()
