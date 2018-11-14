# 实现简单的遗传算法
# 以目标算式 y = 10*sin(5x) + 7*cos(4x)为例，计算其最大值
# 编码方式选取二进制编码方式
# 以轮盘转赌的方式进行选择
# 交换方式选择单点交叉
# 变异方式采用基本位变异，随机挑选一个基因座进行变异
# 限制进化次数


import math
import numpy as np
import random
import matplotlib.pyplot as plt
from time import clock


class simple_genetic():
    def __init__(self, pc, pm, evolution_time, pop_size=500, chrom_length=10, min_value=0, max_value=10):
        self.pop_size = pop_size  # 初始化时候的种群数量
        self.chrom_length = chrom_length  # 染色体的长度，相当于精度的衡量
        self.pc = pc  # 交配概率
        self.pm = pm  # 突变概率
        self.max_value = max_value  # 所允许的基因不超过的最大值
        self.min_value = min_value  # 所允许的基因不超过的最小值
        self.evolution_time = evolution_time # 限定进化次数

    # 种群的初始化
    # 以二进制作为编码方式
    def geneEncoding(self):
        pop_chrom = []
        for i in range(self.pop_size):
            sin_chrom = []
            for i in range(self.chrom_length):
                # 随机将一条染色体二进制编码
                sin_chrom.append(np.random.randint(0, 2))  # 若是使用import random中的random.randint,则是random.randint(0,1)
            pop_chrom.append(sin_chrom)
        return np.array(pop_chrom)

    # 解码
    # 将二进制数转换为十进制数，传入参数为二进制编码的种群
    # 转换为十进制数后再转变为对应区间[min_value, max_value]里的值
    def decoding(self, encoding_list):
        decoding_list = []
        for i in range(len(encoding_list)):
            sin_value = 0
            for j in range(self.chrom_length):
                sin_value += encoding_list[i][j]*math.pow(2, j)
            decoding_list.append(sin_value)
        decoding_list = self.min_value + np.array(decoding_list)*(self.max_value - self.min_value)/(math.pow(2, self.chrom_length)-1)
        return decoding_list

    # 适应度函数，在这里适应值就是函数值，函数值越大适应度越大，因为寻找的是最大值
    def fitness(self, decoding_list):
        fit_value = []
        for i in range(len(decoding_list)):
            fit_value.append(10*math.sin(5*decoding_list[i])+7*math.cos(4*decoding_list[i]))
        return np.array(fit_value)

    # 因为是选择最大值，这里将负值淘汰掉
    def cut_negative(self, fit_value):
        bool_index = (fit_value < 0)
        fit_value[bool_index] = 0  # 将小于0的适应度值设置为0，意味着被淘汰掉, 下一次被选中的概率为0
        return fit_value

    # 计算个体累计适应度值，例如一组数据被选中概率分别为[0.1,0.2,0.3,0.4],则对应累积适应度值为[0.1,0.3,0.6,1]
    def cumsum(self, selection_possibility):
        cum_possibility = []
        t = 0
        for i in range(len(selection_possibility)):
            t += selection_possibility[i]
            cum_possibility.append(t)
        return np.array(cum_possibility)


    # 自然选择，要淘汰掉适应值不好的
    # 这里进行比例选择，即适应值越大的，被选择上的概率就越大
    # pop为二进制编码的种群
    def selection(self, pop, fit_value):
        # 计算适应度值的总加和
        total_fit = sum(fit_value)
        # 按比例选中概率，即轮盘赌法选择
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
        for i in range(len(pop)-1):
            # 判断该次是否作出交配
            if random.random() < self.pc:
                # 随机选中一点进行单点交叉
                rand_cross = np.random.randint(0, self.chrom_length)  # 随机数取值范围为前闭后开区间
                temp1 = pop[i][rand_cross:]
                temp2 = pop[i+1][rand_cross:]
                pop[i][rand_cross:] = temp2
                pop[i+1][rand_cross:] = temp1

    # 变异操作
    def mutation(self, pop):
        for i in range(len(pop)):
            # 判断该次是否需要变异
            if random.random() < self.pm:
                # 随机选择突变的基因位置
                mutation_loc = np.random.randint(0, self.chrom_length)
                # 进行位变异
                if pop[i][mutation_loc] == 1:
                    pop[i][mutation_loc] = 0
                else:
                    pop[i][mutation_loc] = 1

    # 找出最优解以及最优解的基因编码，每生成一个种群时候都要找到它的最优解以及最优编码是多少
    # 返回最佳个体（二进制编码）、二进制解码后实数值、最佳适应度值
    def best(self, pop, fit_value):
        index = fit_value.argmax()  # 适应度最大的索引值
        best_fit = fit_value[index]
        best_individual = pop[index]  # 最佳个体，二进制编码
        # 将个体的二进制编码转换成实数值
        t = 0
        for i in range(len(best_individual)):
            t += best_individual[i]*math.pow(2,i)
        best_t = self.min_value + t*(self.max_value - self.min_value)//(math.pow(2, self.chrom_length)-1)
        return best_individual, best_t, best_fit

    # 算法入口
    def algorithm(self):
        # 种群的初始化
        pop = self.geneEncoding()
        X = np.zeros(self.evolution_time) # 存储每一代的解码后的最优解
        Y = np.zeros(self.evolution_time) # 存储每一代的最优的适应值
        for i in range(self.evolution_time):
            decoding_list = self.decoding(pop)
            fit_value = self.fitness(decoding_list)
            fit_value = self.cut_negative(fit_value)
            best_individual, best_t, best_fit = self.best(pop, fit_value)
            X[i] = best_t
            Y[i] = best_fit
            pop = self.selection(pop, fit_value)
            self.crossover(pop)
            self.mutation(pop)
        return X, Y

    def draw_picture(self):
        X, Y = self.algorithm()
        time = []
        Y.sort()
        for i in range(len(Y)):
            time.append(i)
        plt.plot(time, Y)
        plt.show()


#------------------ test  ---------------------#
t0 = clock()
new = simple_genetic(pc=0.6, pm=0.01,evolution_time = 500)
print(new.algorithm())
new.draw_picture()
t1 = clock()
print('time:' + str(t1-t0))
