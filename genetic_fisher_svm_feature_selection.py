"""
遗传算法+Fisher Score+SVM基于封装式特征提取
"""
import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier


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


class SmoOpt():
    # 主要用来存储SMO优化过程中用到的数据
    def __init__(self, data, label, kTup = ('lin', 1.3), maxtime=1000, thread=0.001, C = 200):
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
        return 0
    kii = opts.kernel[i, i]
    kij = opts.kernel[i, j]
    kjj = opts.kernel[j, j]
    eta = kii + kjj - 2*kij
    if eta <= 0:
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
        return 0
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


# 测试结果
def svm_test(train_data, train_label, test_sample, test_label, maxtime=10000, C=200, thread=0.0001, kTup=('lin', 1.3)):
    # 对训练样本的错误率分析
    train_test = SmoOpt(train_data, train_label, kTup, maxtime, thread, C)
    alpha, b = SmoAlpha(train_test)
    train_test.alpha, train_test.b = alpha, b  # 获得训练样本优化后获得的alpha，和b,并更新到类对象中
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
    # 对测试样本的测试
    kernelEva = KernelTrans(train_test, test_sample)
    predict = np.sum(alpha * train_label * kernelEva) + b
    # 若测试错误，则返回1
    if np.sign(predict) == test_label:
        return 1
    else:
        return 0


# Fisher score特征打分, label为两类
# 返回的score为Series类型
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


class GeneticFisher:
    def __init__(self, data, label, pc=0.01, pm=0.7, evolution_time=500, punish_value=0, mapping_value=0.0, pop_size=500):
        """
        :param data: 数据框形式, 每一行为一个样本，每一个为一个特征
        :param label: 类别标签
        :param pc: 交配概率
        :param pm: 突变概率
        :param mapping_value: 将fisher score映射到[mapping_value, 1-mapping_value]区间上
        :param evolution_time: 限定的进化次数
        :param punish_value: 适应度函数的惩罚参数值（特征过多时候作出一定惩罚值）
        :param pop_size:  初始化种群大小，一般为500
        """
        self.data = data
        self.label = label
        self.pc = pc
        self.pm = pm
        self.mapping_value = mapping_value
        self.evolution_time = evolution_time
        self.punish_value = punish_value
        self.time = 1  #记录遗传算法的迭代次数
        self.chrom_length = self.data.shape[1]  #特征数量
        self.pop_size = pop_size

    # 种群的初始化, 选择初始特征
    def init_population(self):
        score = fisher_score(self.data, self.label)
        # 将fisher 分数映射到mapping_value区间上
        mapping_score = ((1-2*self.mapping_value)/(score.max()-score.min()))*score + \
                        (self.mapping_value*(score.max()+score.min()) - score.min())/(score.max()-score.min())
        pop_chrom = []
        for i in range(self.pop_size):
            sin_chrom = []
            fisher = np.random.uniform(self.mapping_value, 1-self.mapping_value, self.chrom_length)
            for j in range(self.chrom_length):
                feature = self.data.columns[j]
                if mapping_score[feature] > fisher[j]:
                    sin_chrom.append(1)
                else:
                    sin_chrom.append(0)
            pop_chrom.append(sin_chrom)
        # 每一行为一条染色体
        return np.array(pop_chrom)

    # 适应度函数，将分类准确率作为适应度函数。
    # 计算染色体群中每一条染色体的适应值
    def fitness(self, pop):
        fit_value = []
        for i in range(self.pop_size):
            selection_data = self.data[self.data.columns[pop[i] == 1]]  # 选择特征
            # 留一法进行交叉验证（样本数量较少）
            right = 0
            for j in range(selection_data.shape[0]):
                train_data = selection_data.drop(selection_data.index[j])
                train_array = np.array(train_data)
                train_label = np.delete(self.label, j)
                test_sample = np.array(selection_data.iloc[j])
                test_label = self.label[j]
                right += svm_test(train_array, train_label, test_sample, test_label)
            accuracy_rate = right/selection_data.shape[0]  #计算错误率
            sig_fit_value = accuracy_rate - self.punish_value*selection_data.shape[1]
            fit_value.append(sig_fit_value)
        return np.array(fit_value)

    # 计算个体累计适应度值，例如一组数据被选中概率分别为[0.1,0.2,0.3,0.4],则对应累积适应度值为[0.1,0.3,0.6,1]
    def cumsum(self, selection_possibility):
        cum_possibility = []
        t = 0
        for i in range(len(selection_possibility)):
            t += selection_possibility[i]
            cum_possibility.append(t)
        return np.array(cum_possibility)

    # 自然选择，轮盘赌法选择适应值最大的个体
    def selection(self, pop, fit_value):
        total_fit = sum(fit_value)
        selection_possibility = fit_value/total_fit
        cum_possibility = self.cumsum(selection_possibility)  # 计算个体累积适应度
        # 首先生成群体个数个在0-1之间的随机小数
        selection_value = []
        for i in range(len(pop)):
            selection_value.append(random.random())
        selection_value.sort()
        # 开始进行轮盘选择，与个体累积适应度值做比较
        # 例如多累积适应度值为[0.1,0.3,0.6,1], 若生成的随机数为0.2, 则第二个个体被选上。
        fit_in = 0  # fit_in为被拿来比较的的个体编码
        i = 0
        new_pop = pop
        # 该循环的操作就是将随机数确定到各个区间上，然后找到对应的被选中的个体
        while i < len(pop):
            if selection_value[i] < cum_possibility[fit_in]:
                # 说明成功将随机数序列的第i个落入对应区间，并将对应区间代表的个体作为被选中个体
                new_pop[i] = pop[fit_in]
                i += 1
            else:
                fit_in += 1
        return new_pop

    # 进行交配操作
    def crossover(self, pop):
        for i in range(len(pop) - 1):
            # 判断该次是否作出交配
            if random.random() < self.pc + np.log(self.time)/(10*np.log(self.evolution_time)):
                # 随机选中一点进行单点交叉
                rand_cross = np.random.randint(0, self.chrom_length)  # 随机数取值范围为前闭后开区间
                temp1 = pop[i][rand_cross:]
                temp2 = pop[i + 1][rand_cross:]
                pop[i][rand_cross:] = temp2
                pop[i + 1][rand_cross:] = temp1
        return pop

    # 进行突变操作
    def mutation(self, pop):
        for i in range(len(pop)):
            # 判断该条染色体是否需要变异操作
            if random.random() < self.pm + 0.04*np.cos(np.pi*(self.time-1)/2*(self.evolution_time-1)):
                # 随机选择突变的基因位置
                mutation_loc = np.random.randint(0, self.chrom_length)
                # 进行位变异
                if pop[i][mutation_loc] == 1:
                    pop[i][mutation_loc] = 0
                else:
                    pop[i][mutation_loc] = 1
            return pop

    # 寻找该次群体中的最优适应度值以及最优染色体
    def find_best(self, pop, fit_value):
        index = fit_value.argmax()  # 适应度最大的索引值
        best_fit_value = fit_value[index]
        best_chromosome = pop[index]
        return best_fit_value, best_chromosome

    # 算法入口
    def genetic_algorithm(self):
        # 种群的初始化
        pop = self.init_population()
        for i in range(self.evolution_time):
            fit_value = self.fitness(pop)
            preserved_index = np.argsort(-fit_value)[:int(0.2*self.pop_size)]
            preserved_pop = pop[preserved_index]
            changed_pop = pop[np.argsort(-fit_value)[int(0.2*self.pop_size):]]
            changed_pop = self.crossover(changed_pop)
            changed_pop = self.mutation(changed_pop)
            print(changed_pop)
            pop = np.concatenate([preserved_pop, changed_pop])
        fit_value = self.fitness(pop)
        best_fit_value, best_chromosome = self.find_best(pop, fit_value)
        best_features = self.data.columns[best_chromosome == 1]
        return best_features


def main():
    data = pd.read_csv("/Users/laisenying/PycharmProjects/untitled1/p1/genus_relative_PRJNA282013.csv")
    meta = pd.read_csv("/Users/laisenying/PycharmProjects/untitled1/p1/PRJNA282013_meta.csv")
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

    # 过滤掉在ASD和health类超过百分之90的样本都不存在的特征
    data_ASD = data_T.loc[(meta2[meta2['disease'] == 'ASD'])['sample_name']]
    data_health = data_T.loc[(meta2[meta2['disease'] == 'Health'])['sample_name']]
    ASD_selection = (data_ASD == 0).sum() <= (0.9 * data_ASD.shape[0])
    health_selection = (data_health == 0).sum() <= (0.9 * data_health.shape[0])
    data_T = data_T[data_T.columns[ASD_selection | health_selection]]

    label = np.zeros(len(meta2))
    for i in range(len(meta2)):
        if meta2['disease'][i] == 'ASD':
            label[i] = 1
        else:
            label[i] = -1

    genetic_feature = GeneticFisher(data_T, label, pc=0.01, pm=0.7, evolution_time=500,
                                    punish_value=0, mapping_value=0.01, pop_size=500)
    best_features = genetic_feature.genetic_algorithm()
    print(best_features)
    data_selection = data_T[best_features]
    # 遗传算法选出来的特征进行随机森林重要性评估
    random_forest_feature(data_selection, label)


if __name__ == "__main__":
    main()
