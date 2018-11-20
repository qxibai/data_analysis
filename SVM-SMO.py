import numpy as np
from time import clock
import matplotlib.pyplot as plt

'''
一个可以实现二分类的SVM，可选的核函数有高斯核函数、线性、以及多项式
使用SMO算法求出最优的alpha值，基本算法步骤：
1、基于alpha一个初值，然后先遍历所有的数据集合，对每一个不满足KKT条件的alpha i作为第一个要优化的变量alpha i, 当遍历完全部的数据集合后
在非0非C的alpha上进行寻找可进行优化的alpha i,若没有找到，则再次遍历整个数据集合。这里以entireset作为设置是否需要重新进入整个数据集进行查询
2、选定好alpha i后，有三种方式进行alpha j的选择, 若存在非0非C,则优先在非0非C的alpha中进行遍历，以启发式算法查找，查找Ei-Ej绝对值最大的那个点
这样最可能使得alpha j改变的程度最大。若不存在非0非C或者在非0非C中没有找到能够移动足够步长的alpha j，则在非0非C中进行随机选择一次alpha j，若此
次选择的alphaj 依然不能满足提升足够步，则进行第三种选择方式，在全部样本中进行随机选择一次。若三种方法都没有选择出可以优化的alpha j,则重新更换
alpha i
'''


class SmoOpt():
    # 主要用来存储SMO优化过程中用到的数据
    def __init__(self, data, label, kTup, maxtime=1000, thread=0.001, C = 200):
        # 读取是护具
        self.data = data
        self.label = label
        self.C = C  # 软间隔参数C,C越大，非线性拟合的能力就越强
        self.thread = thread
        self.kTup = kTup  # 选择的核函数的类型
        self.m = np.shape(self.data)[0]
        self.alpha = None
        self.b = 0
        self.maxtime = maxtime  # 所能够允许的最大迭代次数
        self.kernel = np.zeros([np.shape(self.data)[0], np.shape(self.data)[0]])
        # 差值矩阵m*2，第一列存放对象的标志位，1表示存在不为0的差值，0表示差值为0, 第二列为实际的差值Ei.
        self.eCache = np.zeros(self.m)


# 计算核函数，输入某一个样本数据xi，返回K(xi,x)的值,即K(xj,xi)构成的列表，j=1,2,...,m
# 注：返回的K中K[j]=k(xj,xi)=kij
# kTup为传入的选择的核函数,例如('lin',k1)
def KernelTrans(opts, xi):
    m, n = np.shape(opts.data)
    k = np.zeros(m)
    if opts.kTup[0] == 'lin':  # 表示使用线性函数，即标准的x·xi类型
        k = np.dot(opts.data, xi)
    elif opts.kTup[0] == 'rbf':  # 选择径向基函数
        for j in range(m):
            k[j] = np.dot(opts.data[j] - xi, opts.data[j] - xi)
        k = np.exp(k/(-1*opts.kTup[1]**2))
    elif opts.kTup[0] == 'poly':    # 多项式函数，输入('poly',p)
        k = (np.dot(opts.data, xi)+1)**opts.kTup[1]
    else:
        raise NameError("the kernel can not be recognized")
    return k


# 计算g(Xi)的值
def predict_gi(opts, alpha_index):
    xi = opts.data[alpha_index]
    k = KernelTrans(opts, xi)
    gi = np.sum(opts.alpha*opts.label*k) + opts.b
    return gi


# 计算Ei值,alpha_index是样本对象的编号
def calE(opts, alpha_index):
    gi = predict_gi(opts, alpha_index)
    Ei = gi - opts.label[alpha_index]
    return Ei


# 计算K11+K22-K12
def caleta(opts, i, j):
    kii = opts.kernel[i, i]
    kjj = opts.kernel[j, j]
    kij = opts.kernel[i, j]
    eta = kii + kjj - 2*kij
    if eta <= 0:  # 不考虑eta小于等于0的情况，这种情况是另外一种解
        print("此时eta值不大于0")
        return 0
    else:
        return eta


# 判断传入进去的alpha是否满足KKT条件
def is_KTT(opts, alpha_index):
    is_ktt = False
    alpha = opts.alpha[alpha_index]
    y_label = opts.label[alpha_index]
    gi = predict_gi(opts, alpha_index)
    if alpha == 0 and y_label*gi >= 1-opts.thread:
        is_ktt = True
    elif alpha == opts.C and y_label*gi <= 1+opts.thread:
        is_ktt = True
    elif 0 < alpha < opts.C and 1 - opts.thread <= y_label*gi <= 1 + opts.thread:
        is_ktt = True
    return is_ktt


# 确定边界L和H
def boundary(opts, i, j):
    if opts.label[i] != opts.label[j]:
        L = max(0, opts.alpha[j] - opts.alpha[i])
        H = min(opts.C, opts.C + opts.alpha[j] - opts.alpha[i])
    else:
        L = max(0, opts.alpha[j] + opts.alpha[i] - opts.C)
        H = min(opts.C, opts.alpha[j] + opts.alpha[i])
    return L, H


# 判断是否前进一步，即是否进行了优化
def takestep(opts, i, j):
    if i == j:
        return 0
    Ei = calE(opts, i)
    Ej = calE(opts, j)
    # 确认边界
    L, H = boundary(opts, i, j)
    if L == H:
        # 若边界相等则停止优化
        print("L==H")
        return 0
    kii = opts.kernel[i, i]
    kij = opts.kernel[i, j]
    kjj = opts.kernel[j, j]
    eta = kii + kjj - 2*kij
    if eta <= 0:
        print("eta<=0")
        return 0
    # 存储旧值
    alpha_oldi = opts.alpha[i]
    alpha_oldj = opts.alpha[j]
    opts.alpha[j] = opts.alpha[j] + opts.label[j]*(Ei-Ej)/eta
    # 进行剪辑，alpha_j的值不能超过边界
    if opts.alpha[j] > H:
        opts.alpha[j] = H
    elif opts.alpha[j] < L:
        opts.alpha[j] = L
    # 判断步长是否有变化
    if abs(opts.alpha[j] - alpha_oldj) < 0.00001:
        opts.eCache[j] = calE(opts, j)
        print('j not move enough')
        return 0
    # 打印能够产生较大移动的j值
    print(j)
    y1 = opts.label[i]
    y2 = opts.label[j]
    # 更新alpha i
    opts.alpha[i] += y1 * y2 * (alpha_oldj - opts.alpha[j])
    # 更新b值
    b1_new = -Ei - y1 * kii * (opts.alpha[i] - alpha_oldi) - y2 * kij * (opts.alpha[j] - alpha_oldj) + opts.b
    b2_new = -Ej - y1 * kij * (opts.alpha[i] - alpha_oldi) - y2 * kjj * (opts.alpha[j] - alpha_oldj) + opts.b
    if 0 < opts.alpha[i] < opts.C:
        opts.b = b1_new
    elif 0 < opts.alpha[j] < opts.C:
        opts.b = b2_new
    else:
        opts.b = (b1_new + b2_new)/2
    # 更新差值矩阵
    opts.eCache[i] = calE(opts, i)
    opts.eCache[j] = calE(opts, i)
    return 1


# 首先判断该alpha i是否需要进行优化，然后对需要优化的alpha i进行alpha j的选择，然后再接着优化alpha j
def optimize(opts, i):
    # 判断是否满足KKT条件，若满足，则不需要进行优化
    is_ktt = is_KTT(opts, i)
    if is_ktt:
        return 0
    else:  # j的选择
        Ei = calE(opts, i)
        opts.eCache[i] = Ei
        is_nonbounds = (opts.alpha > 0)*(opts.alpha < opts.C)  # 返回布尔值
        number_0C = len(opts.alpha[is_nonbounds])
        sort_0C = np.argsort(~is_nonbounds)
        # 第一种选择j的方式
        # 若存在非0又非C的点，则现在非0又非C的点上以启发式算法寻找，看是否能使的其前进
        if number_0C:
            nonbounds = sort_0C[:number_0C]  # 对索引排序，非边界值排在前面
            maxdelta = -1
            maxj = 0
            for j in nonbounds:
                if abs(opts.eCache[j] - Ei) > maxdelta:
                    maxj = j
                    Ej = opts.eCache[j]
                    maxdelta = abs(Ej - Ei)
            if takestep(opts, i, maxj):
                return 1
        # 第二种选择j的方式
        # 遍历所有的边界上的点，即alpha为0或者C的点，随机寻找一个j进行优化
        bounds = sort_0C[number_0C:]
        number_bounds = len(bounds)
        j = i
        while j == i:
            j = bounds[np.random.randint(number_bounds)]
        if takestep(opts, i, j):
            return 1
        # 第三种选择j的方式
        # 对整个数据集随机挑选一个样例作为j
        j = i
        while j==i :
            j = np.random.randint(opts.m)
        if takestep(opts, i, j):
            return 1
    # 若以上三种alpha j的选择方式都不能使得优化前进，则返回0
    return 0


# SMO算法的主体部分
# 传入的opts为类对象
def SmoAlpha(opts):
    # 初始化SMO参数h
    m, n = np.shape(opts.data)
    # alpha和b的初始化
    opts.alpha = np.zeros(m)
    opts.b = 0
    for i in range(m):
        opts.eCache[i] = calE(opts, i)
    for i in range(m):
        opts.kernel[:, i] = KernelTrans(opts, opts.data[i])
    time = 0  # 用于存储优化的次数,初始值为0
    entireset = True  # 用来设置是否要对整个样本进行遍历
    change = 0  # 用于记录进行两变量优化次数
    alphachanged = 0
    while (alphachanged > 0) or entireset:
        # 当alphachanged=0以及entireset为False【表明alphachanged为0时候，又遍历了整个训练样本，仍然是0】
        # 表明alpha不再进行什么改变，此时可以结束优化
        # 最开始不知道哪些是违反KKT的点，先遍历一次整个样本，将所有违反KKT的alpha作为要进行优化的第一个变量
        # 所以最开始设置entireset为True,表明先遍历一次整个训练样本
        alphachanged = 0  # 用于记录遍历整个数据集是否有可优化的alpha，或者遍历边界向量的时候是否有可优化的alpha
        if entireset:
            # 进行外层循环，选取违反KKT条件的样本点，并将其对应的变量alpha作为第一个变量a1
            for i in range(m):
                alphachanged += optimize(opts, i)  # 记录优化次数
            time += 1  # 迭代次数加1
            change += alphachanged
        else:  # 遍历所有非边界样本点
            nonBounds = np.nonzero(opts.alpha)[0]
            for i in nonBounds:
                alphachanged += optimize(opts, i)
            time += 1  # 迭代次数加1
            change += alphachanged
        if entireset:
            entireset = False  # 遍历完一次整个样本之后，不需要再整个样本的遍历，只需要遍历非边界样本点
        elif alphachanged == 0:  # 若遍历非边界样本没怎么改变alpha值，则对整个训练集进行遍历
            entireset = True
        if time > opts.maxtime:
            print("已经达到最大允许的迭代次数，时间到")
            break
    print("迭代次数：" + str(time))
    print("优化次数" + str(change))
    return opts.alpha, opts.b


# 读取数据
def txt_to_mat(filename):
    fr = open(filename)
    lines = fr.readlines()
    data = []
    label = []
    for line in lines:
        line = line.strip()
        linetolist = line.split(',')
        data.append([float(linetolist[0]),float(linetolist[1])])
        label.append(float(linetolist[2]))
    data = np.array(data)
    return data, label


# 训练分类数据的可视化
def txt_to_picture(data, label):
    color_label = []
    for i in range(len(label)):
        if label[i] == 1:
            color_label.append('red')
        else:
            color_label.append('green')
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], color=color_label)


# 预测分类数据的可视化
def test_result_picture(data, label, error_index):
    color_label = []
    for i in range(len(label)):
        if label[i] == 1:
            color_label.append('red')
        else:
            color_label.append('green')
    for x in error_index:
        color_label[x] = 'black'
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], color=color_label)


# 测试结果
def test(maxtime, C, thread, kTup, train_data, train_label, test_data, test_label):
    # 训练样本的可视化
    txt_to_picture(train_data, train_label)
    plt.title("Original training set")
    plt.show()
    # 测试样本的可视化
    txt_to_picture(test_data, test_label)
    plt.title("Original testing set")
    # 对训练样本的错误率分析
    train_test = SmoOpt(train_data, train_label, kTup, maxtime, thread, C)
    alpha, b = SmoAlpha(train_test)
    train_test.alpha, train_test.b = alpha, b # 获得训练样本优化后获得的alpha，和b,并更新到类对象中
    print("b的值为："+ str(b))
    # 获得支持向量的个数
    # 判断alpha中有多少个在0到C之间，代表有多少个支持向量
    k = len(alpha[(alpha < C) * (alpha > 0)])
    print("there are %d Support Vectors in testing set" % k)
    error_count = 0
    error_train_index = []
    # 对训练样本集合的测试
    for i in range(train_test.m):
        predict = predict_gi(train_test, i)
        if np.sign(predict) != train_test.label[i]:
            error_count += 1
            error_train_index.append(i)
    print("训练集样本中的错误率为：%f" % (error_count*100/train_test.m) + "%")
    # 对训练集样本做结果可视化，黑色的点代表错误分类的点
    test_result_picture(train_data, train_label, error_train_index)
    plt.title("result of training test")
    plt.show()
    # 对测试样本的测试
    testing_test = SmoOpt(test_data, test_label, kTup, maxtime, thread, C)
    error_count = 0
    error_test_index = []
    for i in range(testing_test.m):
        # 计算测试样本对象i的映射
        xi = testing_test.data[i]
        kernelEva = KernelTrans(train_test, xi)   # 注意这里K(x,xi)中的x是指训练集里的数据
        predict = np.sum(alpha*train_label*kernelEva) + b  # 其实这里可以直接用支撑向量来激素哪，因为其他的量都是0
        if np.sign(predict) != testing_test.label[i]:
            error_count += 1
            error_test_index.append(i)
    print("测试集样本中的错误率为：%f" % (error_count*100 / testing_test.m)  + "%")
    # 对测试集样本做结果可视化，黑色的点代表错误分类的点
    test_result_picture(test_data, test_label, error_test_index)
    plt.title("result of testing test")
    plt.show()


if __name__ == "__main__":
    t0 = clock()
    # 读取训练和测试样本文件
    train_data, train_label = txt_to_mat("train.txt")
    test_data, test_label = txt_to_mat("testing.txt")
    # 设置参数
    '''
    maxtime: 设置所允许的最大迭代次数
    C: 惩罚参数
    thread: 容错率，即满足KKT条件容忍度
    kTup:核函数类型，可以自己定义
    '''
    maxtime = 10000
    C = 200
    thread = 0.0001
    kTup = ('rbf', 1.3)
    # 测试与训练
    test(maxtime, C, thread, kTup, train_data, train_label, test_data, test_label)
    t1 = clock()
    print("time:" + str(t1 - t0))
