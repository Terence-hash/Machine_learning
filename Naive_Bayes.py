import numpy as np
import pandas as pd
import math
import random
from copy import deepcopy


# 创建三维列表
def makeList(targetNum, featureNum, n):
    inner_ls = []
    outer_ls = []
    for j in range(featureNum):
        inner_ls.append(deepcopy([1] * n))
    for i in range(targetNum):
        outer_ls.append(deepcopy(inner_ls))
    return outer_ls


# 在不覆盖第一行的情况下插入列标签
def addColumns(df, label):
    d1 = list(df.columns)
    row1 = [float(d1[0]), float(d1[1]), float(d1[2]), float(d1[3]), d1[4]]  # 更改数据类型
    df1 = pd.DataFrame(row1).T
    df1.columns = label
    df.columns = label
    df = pd.concat([df1, df], axis=0, ignore_index=True)
    return df


# 列表中数据取对数之后的和
def logSumList(lst):
    res = 0
    for i in lst:
        res += math.log(i, 10)
    return res


# 计算正态分布的概率
def gaussFunc(x, u, var):
    prob = 1 / math.sqrt(2 * math.pi * var) * math.exp(-math.pow(x - u, 2) / (2 * var))
    return prob


# 随机划分数据集,也可以用pandas自带的sample()函数进行抽样
def dataPartition(src_data, test_ratio):
    # random.seed(111)
    flower_group = src_data.groupby("label")
    fg = [0] * 3
    test = [0] * 3
    train = [0] * 3
    for i in range(3):
        fg[i] = flower_group.get_group(i)
        total_length = len(fg[i])
        test_length = total_length * test_ratio
        allRow = [x for x in range(total_length)]
        testRow = []
        while len(testRow) < test_length:
            ranRow = random.randint(0, total_length - 1)
            if ranRow not in testRow:
                testRow.append(ranRow)
        trainRow = list(set(testRow) ^ set(allRow))
        test[i] = fg[i].iloc[testRow, :]
        train[i] = fg[i].iloc[trainRow, :]
    testdata = pd.concat([test[0], test[1], test[2]], ignore_index=True)  # 测试集
    traindata = pd.concat([train[0], train[1], train[2]], ignore_index=True)  # 训练集
    return traindata, testdata


# 读取文件
def fileRead(filename, testRatio):
    df = pd.read_csv(filename)
    df = addColumns(df, [0, 1, 2, 3, 'label'])
    df.label = df.label.map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
    # print('_' * 35, "查看缺失值", '_' * 35)
    # print(df.isnull().sum())
    # df = df.dropna()  # 处理缺失值
    # group = df.groupby(['label'])
    # print(group.groups)
    data_train, data_test = dataPartition(df, testRatio)
    return data_train, data_test


# 对训练样本进行处理获得先验概率和高斯分布的参数
def processing(traindata):
    flower = traindata.groupby(['label'])

    flower_dic = dict(flower.size())
    flowerType = list(flower_dic.keys())  # 花类型
    priorPro = [1 / (len(traindata) + 2)] * 3  # 为先验概率预设列表
    for i in flowerType:
        # if i in flower_dic:
        priorPro[i] = (flower_dic[i] + 1) / (len(traindata) + 2)
    # print(priorPro)

    dataCol = list(traindata.columns[:-1])  # 列标签
    para_ls = makeList(3, 4, 2)  # para_ls为存储数学期望和方差的预设列表
    for ft in flowerType:
        fl = flower.get_group(ft)
        for dc in dataCol:
            dt = fl[dc]
            para_ls[ft][dc][0] = dt.mean()  # 数学期望
            para_ls[ft][dc][1] = dt.var()  # 方差
    return priorPro, para_ls


# 对测试集中的鸢尾花类型进行预测
def predict(filename, testratio):
    trainData, testData = fileRead(filename, testratio)
    priorProb, paraList = processing(trainData)

    target = np.array(testData['label'])  # 测试集中花种类的真实值
    test = testData.drop(['label'], axis=1)
    array = np.array(test)
    res = []  # 存放预测结果
    accuracy_sum = 0
    for rowIndex in range(len(array)):
        feature = [0] * 4
        for f in range(4):
            feature[f] = array[rowIndex][f]

        pls = makeList(3, 4, 1)  # 为正态分布的概率预设列表
        for i in range(3):
            for j in range(4):
                u = paraList[i][j][0]
                var = paraList[i][j][1]
                pls[i][j] = gaussFunc(feature[j], u, var)

        # p0, p1, p2分别为花0,1,2的概率
        p0 = logSumList(pls[0]) + math.log(priorProb[0], 10)
        p1 = logSumList(pls[1]) + math.log(priorProb[1], 10)
        p2 = logSumList(pls[2]) + math.log(priorProb[2], 10)
        maxProb = max(p0, p1, p2)
        if p0 == maxProb:
            res.append('Iris-setosa')
            fcode = 0
        elif p1 == maxProb:
            res.append('Iris-versicolor')
            fcode = 1
        else:
            res.append('Iris-virginica')
            fcode = 2
        if fcode == target[rowIndex]:
            accuracy_sum += 1
    accuracy_rate = accuracy_sum / array.shape[0]
    return accuracy_rate, res, testData


# 将测试结果存入iris_predict.csv文件
accRate, result, data_reserve = predict("iris.data", 0.2)
# print(result)
print("精确度为{:.4f}%".format(accRate * 100))
columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
data_reserve.columns = columns
data_reserve['predict'] = result
data_reserve.to_csv('iris_predict.csv', sep=',', index=False)
