import numpy as np

# 一元回归分析
class SingleRegression():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    # 求解线性回归方程的斜率
    def coef(self):
        return np.cov(X,Y)[0][1]/np.var(X, ddof=1)

    # 求解线性回归方程截距
    def intercept(self):
        return np.mean(Y)-self.coef()*np.mean(X)

    # 预测
    def predict(self, x):
        return self.coef()*x + self.intercept()

    # 计算损失函数
    def cost(self):
        return(np.mean((self.predict(self.X)-self.Y)**2))

    # 计算拟合效果
    def pearson(self):
        return np.cov(X,Y)[0][1]/(np.std(Y,ddof=1)*np.std(X,ddof=1))

    def sklearn_score(self):
        SS_t = sum((self.Y-np.mean(Y))**2)
        y2 = map(lambda x: self.predict(x), X)
        Y2 = []
        for y in y2:
            Y2.append(y)
        SS_r = sum((self.Y-np.array(Y2))**2)
        return 1 - SS_r/SS_t

    # 入口
    def result(self):
        print("线性回归方程的系数："+str(self.coef()))
        print("线性回归方程的截距："+str(self.intercept()))
        print("拟合的线性方程为：y="+str(self.coef())+"x+"+str(self.intercept()))
        print("皮尔逊相关系数为："  + str(self.pearson()))
        print("sklearn_score中的R方为：" + str(self.sklearn_score()))
 

from numpy.linalg import *
import matplotlib.pyplot as plt


class MultiLinearRegression():
    def __init__(self, X, y, learningrate=None, loop = None):
        self.X = X
        self.y = y
        self.learningRate = learningrate
        self.loop = loop

    # 计算损失函数
    def loss_function(self, weight, baise):
        predict_y = np.dot(self.X, weight.T) + baise
        loss = sum((predict_y - self.y)**2)/self.X.shape[0]
        return loss

    # 直接求得weight和b,最小二乘法
    def least_square(self):
        a = np.ones([len(self.X),1])
        x = np.concatenate([a, self.X], axis=1)
        try:
            w = np.dot(inv(np.dot(x.T, x)), x.T)
            weight = np.dot(w, self.y)
        except LinAlgError:
            print("不适合用最小二乘法直接求权系数")
        else:
            baise = weight[0]
            return weight[1:], baise

    # 梯度下降法求解损失函数最小化
    # 这里将b合并到weight中，相当于多了一个x0=1与w0
    # 如果样本量过大的时候可以采取随机梯度下降法
    def gradient(self):
        a = np.ones([len(self.X), 1])
        x = np.concatenate([a, self.X], axis=1)
        # 初始化weight和baise
        weight = np.ones(self.X.shape[1]+1)
        loss = self.loss_function(weight[1:], weight[0])
        loss_list = []
        loss_list.append(loss)
        for i in range(self.loop):
            weight_gradient = np.dot(x.T, np.dot(x, weight)-self.y)*2/len(self.X)
            weight = weight - self.learningRate*weight_gradient
            loss = self.loss_function(weight[1:], weight[0])
            loss_list.append(loss)
            if weight_gradient.any() == 0:
                break
        baise = weight[0]
        weight = weight[1:]
        # 损失函数画图绘制
        loss_list.sort()
        loss_list.reverse()
        time = []
        for i in  range(len(loss_list)):
            time.append(i)
        plt.plot(time, loss_list)
        plt.title("loss curve")
        plt.show()
        return weight, baise

    #  y = W*X + b; w0x0 + w1x1 + w3x2+... + 得到预测值
    def least_square_predict(self, x_test):
        ##最小二乘法
        weight, baise = self.least_square()
        print("最小二乘法方法计算得损失函数值为：" + str(self.loss_function(weight, baise)))
        print("最小二乘法求得的权值为"+ str(weight))
        return np.dot(x_test, weight.T) + baise

    def gradient_predict(self, x_test):
        # 梯度下降法求解
        weight, baise = self.gradient()
        print("梯度下降方法计算得损失函数值为：" + str(self.loss_function(weight, baise)))
        print("梯度下降法法求得的权值为" + str(weight))
        return np.dot(x_test, weight.T)


    # 打印输出
    def output(self, x_test, y_test):
        print("用最小二乘法求解多元线性回归：")
        y_predict = self.least_square_predict(x_test)
        for i in range(len(x_test)):
            print('Predicted: %s, Target: %s' % (y_predict[i], y_test[i]))
        print("用梯度下降法求解：")
        y_predict = self.gradient_predict(x_test)
        for i in range(len(x_test)):
            print('Predicted: %s, Target: %s' % (y_predict[i], y_test[i]))


# 测试数据
if __name__="__main__":
    X = np.array([6, 8, 10, 14, 18])
    Y = np.array([7, 9, 13, 17.5, 18])
    new = SingleRegression(X, Y)
    new.result()

    X = np.array([[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]])
    y = np.array([7, 9, 13, 17.5, 18])
    X_test = np.array([[8,2], [9, 0], [11, 2], [16, 2], [12, 0]])
    y_test = [11, 8.5, 15, 18, 11]
    new = MultiLinearRegression(X, y, 0.00001, 1000)
    new.output(X_test,  y_test)
