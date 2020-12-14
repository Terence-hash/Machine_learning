"""
 *   @file：     KNN.py
 *   @date：     2020-10-7
 *   @brief：    KNN算法的测试
 *   @Author：   Terence Tan
 *------------------------------------
"""
from numpy import *
from random import randint
import matplotlib.pyplot as plt
import matplotlib

# 图表中显示中文
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

"""
 funtion:根据欧式距离对数据进行分类
 parameter：inX,输入测试数据;dataSet,训练数据集合;labels,训练数据标签;k,近邻点个数
 return:与测试数据最接近的训练数据的标签
"""


def classify0(inX, dataSet, labels, k):
    # 计算欧式距离
    dataSetSize = dataSet.shape[0]  # 矩阵dataSet的行数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5

    sortedDistIndicies = distances.argsort()  # 将数据从小到大进行排序
    classCount = {}
    # 确定与输入点距离最小的k个点的分类
    for i in range(k):
        # 将相距较近的k个点放入字典中，键值对中的值为相应类别出现的次数
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)  # 按照每类点出现的次数由大到小进行排序
    return sortedClassCount[0][0]  # 返回字典中第一个键值对的键，即返回前k个点出现次数最多的类别（标签）


"""
 funtion:读取文件并使数据可视化
 parameter：filename,文件名
 return:文件数据集，文件数据的标签向量
"""


def file2matrix(filename):
    fr = open(filename)  # 打开文件，创建一个file对象
    arrayOLines = fr.readlines()  # 返回包含所有行的一个列表
    while '\n' in arrayOLines:  # 删除独立的空行
        arrayOLines.remove('\n')
    numberOfLines = len(arrayOLines)  # 返回列表的长度，即源文件的行数
    returnMat = zeros((numberOfLines, 4))  # 创建返回的矩阵

    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()  # 去掉字符串左边和末尾的空格，但不能去掉独立的空行
        listFromLine = line.split(',')  # 以','为分隔符分隔字符串
        returnMat[index, :] = listFromLine[0: 4]
        classLabelVector.append(listFromLine[-1])  # 将列表的最后一列（标签）存储到向量中
        index += 1

    return returnMat, classLabelVector


def depict(Mat, labels):
    # 原始数据可视化
    colors = ['r' if i == 'Iris-setosa' else i for i in labels]
    colors = ['g' if i == 'Iris-versicolor' else i for i in colors]
    colors = ['b' if i == 'Iris-virginica' else i for i in colors]
    # print(colors)
    lenOfMat = len(Mat)

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    for index in range(lenOfMat):
        if colors[index] == 'r':
            p11 = ax1.scatter(Mat[index, 0], Mat[index, 3], c=colors[index], s=15)
        if colors[index] == 'g':
            p12 = ax1.scatter(Mat[index, 0], Mat[index, 3], c=colors[index], s=15)
        if colors[index] == 'b':
            p13 = ax1.scatter(Mat[index, 0], Mat[index, 3], c=colors[index], s=15)
    plt.legend((p11, p12, p13), ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'))
    plt.title("第一个与第四个特征")
    plt.xlabel("特征一")
    plt.ylabel("特征四")
    plt.grid(True, which='major')

    ax2 = fig.add_subplot(312)
    for index in range(lenOfMat):
        if colors[index] == 'r':
            p21 = ax2.scatter(Mat[index, 1], Mat[index, 3], c=colors[index], s=15)
        if colors[index] == 'g':
            p22 = ax2.scatter(Mat[index, 1], Mat[index, 3], c=colors[index], s=15)
        if colors[index] == 'b':
            p23 = ax2.scatter(Mat[index, 1], Mat[index, 3], c=colors[index], s=15)
    plt.legend((p21, p22, p23), ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'))
    plt.title("第二个与第四个特征")
    plt.xlabel("特征二")
    plt.ylabel("特征四")
    plt.grid(True, which='major')

    ax3 = fig.add_subplot(313)
    for index in range(lenOfMat):
        if colors[index] == 'r':
            p31 = ax3.scatter(Mat[index, 2], Mat[index, 3], c=colors[index], s=15)
        if colors[index] == 'g':
            p32 = ax3.scatter(Mat[index, 2], Mat[index, 3], c=colors[index], s=15)
        if colors[index] == 'b':
            p33 = ax3.scatter(Mat[index, 2], Mat[index, 3], c=colors[index], s=15)
    plt.legend((p31, p32, p33), ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'))
    plt.title("第三个与第四个特征")
    plt.xlabel("特征三")
    plt.ylabel("特征四")
    plt.grid(True, which='major')
    return plt
    # plt.savefig("scatter.png", dpi=120)


"""
 funtion:归一化数值
 parameter：dataSet
 return:归一化后的数据集，源数据取值范围，源数据最小值
"""


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    # normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))  # 源数据当前值减去最小值
    normDataSet = normDataSet / tile(ranges, (m, 1))  # 除以取值范围
    return normDataSet, ranges, minVals


"""
 funtion:测试KNN算法
 parameter：无
 return:无
"""


def test():
    hoRatio = 0.2  # 测试集所占比例
    dataMat, dataLabels = file2matrix('iris.data')  # 读取文件
    srcImg = depict(dataMat, dataLabels)  # 原始数据可视化
    srcImg.show()
    normMat, ranges, minVals = autoNorm(dataMat)  # 归一化处理
    numOfLines = normMat.shape[0]  # (归一化后的)矩阵的行数
    numTestVecs = int(numOfLines * hoRatio)  # 计算测试向量的数量(训练集的20%)
    errorCount = 0.0  # 出错率
    count = 0
    ls = []  # 创建测试数据集
    result = []
    resultLabels = []  # 测试数据的识别结果存储在resultLabels列表中
    while count < numTestVecs:
        # 随机选取训练集中20%互不重复的数据
        i = randint(0, 149)
        if i not in ls:
            ls.append(i)
            # normMat[i, :]为测试集中的一个测试样本
            # 将测试数据与训练集中除去前测试集行数个数据后的剩余数据进行比较
            classifierResult = classify0(normMat[i], normMat[numTestVecs: numOfLines], dataLabels[numTestVecs: numOfLines], 2)
            result.append(normMat[i])
            resultLabels.append(classifierResult)
            print("the classifier came back with: {}, the real answer is : {}".format(classifierResult, dataLabels[i]))
            if classifierResult != dataLabels[i]: errorCount += 1.0
            count += 1
    result = array(result)  # 将列表类型转化为数据类型
    dstImg = depict(result, resultLabels)    # 分类结果可视化
    dstImg.show()
    print("the total error rate is {:%}".format((errorCount / float(numTestVecs))))


# for j in range(10):
test()
