# 1、计算输入的多维度数组各个维度的方差，以方差最大的那个坐标轴为基准，将所有实例的中位数作为切分点，划分为左右两个子区域
# 2、再对划分好的左右两个子区域重复上述操作

import numpy as np

# kd树
class KdNode():
    def __init__(self, split_axis, node, left=None, right=None):
        self.split_axis = split_axis  # 进行切割维度的序号
        self.node = node  # k维向量结点
        self.left = left  # 该结点分割超平面左子空间构成的Kd_tree
        self.right = right  # 该结点分割超平面右子空间构成的Kd_tree

class KdTree():
    def __init__(self, data):
        # k为数据维度
        self.k = len(data[0])

    # 计算输入数据各个维度的方差
    def variance(self, select_data):
        var_ar = select_data.var(axis = 0)
        return var_ar

    # 计算输入数据的中位数(传入的数据是经过排序后的一维数组)
    # 返回数组中位数或者最靠近中位数的索引
    def median(self, sorted_data):
        n = len(sorted_data)
        data_index = {}
        for i in range(len(sorted_data)):
            data_index[sorted_data[i]] = i
        median = np.median(sorted_data)
        median_1 = sorted_data[n//2]
        median_2 = sorted_data[n//2 - 1]
        if n % 2 == 1:
            med_index = data_index[median]
        else:
            if abs(median - median_1) >= abs(median - median_2):
                med_index = data_index[median_2]
            else:
                med_index = data_index[median_1]
        return med_index

    def creatKd(self, new_data):
        if not new_data.any(): # 数据集为空
            return None
        # 求得输入数据方差最大的维度
        axis_maxid = self.variance(new_data).argmax()
        # 按照指定axis列对数组进行排序
        data_sort = new_data[new_data[:, axis_maxid].argsort()]
        # 获得输入数据的中位数的索引，之后以该中位数作为切分点,将超平面划分
        med_id = self.median(data_sort[:, axis_maxid])
        # 获得作为切分点的数据向量
        median_data = data_sort[med_id]
        return KdNode(axis_maxid,
                      median_data,
                      self.creatKd(data_sort[:med_id]),
                      self.creatKd(data_sort[med_id+1:]))

# KdTree的前序遍历
def preorder(root):
    if root is None:
        return
    else:
        print(root.node, root.split_axis)
        preorder(root.left)
        preorder(root.right)

################################################### test_data #################################################
data = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
kd1 = KdTree(data)
Kdtree1 = kd1.creatKd(data)
preorder(Kdtree1)
