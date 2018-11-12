# 利用Kd树搜索最近邻点
# 本程序用40万数据训练时候就耗时30秒，消耗时间巨大。

from build_KdTree import *
import numpy as np
from collections import namedtuple
from math import sqrt

# 定义一个namedtuple类型，分别存放最近坐标点，最近距离和访问过的节点数
result = namedtuple('result', ["nearest_point","nearest_dist","is_continue"])

# 利用Kd树找到输入数据的最近邻点
class KdTreeSearch():
    def __init__(self, Kdtree, point):
        self.Kdtree = Kdtree
        self.point = point
        self.node_visited = 0  # 为结点所在的层次数，树的根结点的的层次数为1
        self.search_path = []
        self.further_path = []  # 保存同一父结点的子结点（即未被访问的结点）

    # 计算当前最近点与目标点之间的距离
    def distance(self, nearest_data):
        return sqrt(np.power(nearest_data - self.point, 2).sum())

    # 首先找到叶结点(找到包含目标点的叶结点）
    # 从根结点依次向下递归
    # 对输入的树进行判断，数据是属于子左子树还是子右子树
    def travel(self, Kdnode):
        self.search_path.append(Kdnode)  # 保存搜索路径
        axis_split = Kdnode.split_axis  # 判断输入的结点是根据哪一个维度进行划分的
        thread = Kdnode.node[axis_split] # 输入的树的根结点上保存的实例在特定维度上的值
        # 对输入的数据进行判断属于左区域还是右区域，将判断的区域作为下一次的输入
        # 同时还要判断左子区域或者右子区域是否为None
        # 将不访问的结点储存在further_path中，此例子中[4,7]没有与其拥有共同父结点的子结点，所以储存到further_path中的为None
        if self.point[axis_split] <= thread:
            if Kdnode.left is not None:
                self.further_path.append(Kdnode.right)
                return self.travel(Kdnode.left)
        else:
            if Kdnode.right is not None:
                self.further_path.append(Kdnode.left)
                return self.travel(Kdnode.right)

    # 传入结点(current_node为当前最近点的父结点），可以根据self.search_path.pop()传入
    # 判断当前最近点的父结点是否比当前最近点距离目标点还要近，若是，则更新最近点与超球体半径
    # 判断是否要根据以current_node为父结点的另一个子结点进行操作
    # 即current_node所在的边界若是与超球体区域相交的话，则继续对其另一个子结点所在区域进行查找，若不相交，则不必进行查找，所有操作停止
    # near_node为当前的最近结点
    def decide_update(self, near_node, current_node,current_further_node, max_dist):
        # 判断current_node的区域与超球体是否相交,若不相交，则什么也不进行操作,否则计算该结点上存储的实例与目标点的距离
        # continue设置为0表示已找到最近点，之后可不再进行操作
        if abs(current_node.node[current_node.split_axis]-self.point[current_node.split_axis]) > max_dist:
            return result._make([near_node, max_dist, 0])
        else:
            dist = self.distance(current_node.node)  # 计算当前结点保存的实例与目标点之间的距离
            if dist < max_dist:
                max_dist = dist  # 更新超球体半径
                near_node = current_node  # 更新当前最近点
            # 继续对共同父亲的子结点进行判断，首先要判断是否存在具有共同父的子结点
            temp = self.update_node(current_further_node, max_dist, near_node)
            max_dist = temp.nearest_dist
            near_node = temp.nearest_point
            return result._make([near_node, max_dist, 1])

    # 判断输入的子结点对应的区域内是否包含更近的点
    # 返回由near_node与max_dist构成的namedtuple数据类型
    def update_node(self, further_node, max_dist, near_node):
        if further_node is None:   # 首先要判断输入的子结点是否为空
            return result._make([near_node, max_dist, 1])
        else:
            dist = self.distance(further_node.node)
            if dist < max_dist:
                max_dist = dist
                near_node = further_node
            if further_node.left is not None:
                return self.update_node(further_node.left, max_dist, near_node)
            if further_node.left is not None:
                return self.update_node(further_node.right, max_dist, near_node)
            return result._make([near_node, max_dist, 1])

    def find_nearest(self):
        # 初始化最近距离以及最近点，即将搜索路径中的最后一个结点上保存的实例作为最近点，其与目标点的距离作为最近距离，即超球体半径
        self.travel(self.Kdtree)  # 搜索包含目标点的叶结点区域
        near_node = self.search_path.pop()  # 将访问路径中的最后一个叶结点作为当前最近点
        max_dist = self.distance(near_node.node)  # 计算当前最近点与目标点的距离,并设置为最佳距离
        while len(self.search_path):
            best_result = self.decide_update(near_node, self.search_path.pop(), self.further_path.pop(), max_dist)
            if best_result.is_continue == 0:
                break
            else:
                max_dist = best_result.nearest_dist
                near_node = best_result.nearest_point
        return near_node.node, max_dist

#########--------------------test----------------------#############

if __name__ == "__main__":
    data = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    kd1 = build_KdTree(data)
    Kdtree1 = kd1.creatKd(data)  # 根据data创建好的Kd树
    # print(Kdtree1.node)
    preorder(Kdtree1)
    point = np.array([3, 4.5])
    near = KdTreeSearch(Kdtree1, point)
    nearest_point, distance = near.find_nearest()
    print("点"+str(point)+"的最近邻点为"+str(nearest_point))
    print("最近邻点到目标点的距离为："+ str(distance))

    from time import clock
    from random import random

    # 产生一个k维随机向量，每一维分量值在0～1之间
    def random_point(k):
        return [random() for _ in range(k)]

    # 产生n个k维随机向量
    def random_points(k, n):
        return [random_point(k) for _ in range(n)]

    N = 40000  # 一共有400000样本
    t0 = clock()
    data = np.random.rand(N, 3)
    print(data)
    kd2 = build_KdTree(data)   # 构建包含四十万个3维空间样本点的kd树
    KdTree2 = kd2.creatKd(data)
    point = np.array([0.1, 0.5, 0.8])
    ret2 = KdTreeSearch(KdTree2, point)
    t1 = clock()
    print("time:"+str(t1-t0))
    print(ret2.find_nearest())
