import numpy as np
import pandas as pd
import math
from copy import deepcopy


# 读取csv文件并显示文件信息
def fread(filename):
    fr = pd.read_csv(filename)
    # print(fr)
    # print(fr.info())
    # print(fr.describe())
    return fr


data_name = ['训练集', '测试集']
data_train = fread('train.csv')
data_test = fread('test.csv')
# data_integrate = [data_train, data_test]  # 将两个数据集整合以便于操作

# 首先去掉对存活与否影响不大的特征
data_train = data_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
data_test = data_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
data_integrate = [data_train, data_test]

# 显示缺失值
# for i in range(2):
# print('_' * 35, data_name[i] + "缺失值", '_' * 35)
# data = data_integrate[i]
# print("Age缺失值总数:{0}\n  占比{1:.2%}".format(data.Age.isnull().sum(), data.Age.isnull().sum() / len(data.Age)))
# print("Fare缺失值总数:{0}\n  占比{1:.2%}".format(data.Fare.isnull().sum(), data.Fare.isnull().sum() / len(data.Fare)))
# print("Embarked缺失值总数:{0}\n  占比{1:.2%}".format(data.Embarked.isnull().sum(),
#                                               data.Embarked.isnull().sum() / len(data.Embarked)))

# 将Sex转换为数值型
for d in data_integrate:
    d.Sex = d.Sex.map({'male': 1, 'female': 0})

# 为不同的Sex与Pclass的组合预分配数组
ages_predict = np.zeros((2, 3))

# 根据Sex和Pclass的组合将Age缺失值赋值为对应的中位数
for d in data_integrate:
    for i in range(0, 2):
        for j in range(0, 3):
            age_predict = d[(d.Sex == i) & (d.Pclass == j + 1)].Age.dropna().median()
            ages_predict[i, j] = int(age_predict + 0.25)
    for i in range(0, 2):
        for j in range(0, 3):
            d.loc[(d.Sex == i) & (d.Pclass == j + 1) & (d.Age.isnull()), 'Age'] = ages_predict[i, j]
    d.Age = d.Age.astype(int)

# 划分五个年龄段
data_train['Age_group'] = pd.cut(data_train.Age, 5)  # 确定分段边界
for d in data_integrate:
    d.loc[(d.Age < 16), 'Age'] = 0
    d.loc[(d.Age >= 16) & (d.Age < 32), 'Age'] = 1
    d.loc[(d.Age >= 32) & (d.Age < 48), 'Age'] = 2
    d.loc[(d.Age >= 48) & (d.Age < 64), 'Age'] = 3
    d.loc[(d.Age >= 64), 'Age'] = 4

# 完成Age的分段赋值后删除Age_group特征
data_train = data_train.drop(['Age_group'], axis=1)
data_integrate = [data_train, data_test]

# 创建Families特征，表示亲人朋友总数
for d in data_integrate:
    d['Families'] = d.SibSp + d.Parch

# 创建Alone特征，表示是否孤身一人
for d in data_integrate:
    d['Alone'] = 0
    d.loc[d.Families == 0, 'Alone'] = 1

# 去掉SibSp, Parch和Families列，保留Alone特征
data_train = data_train.drop(['SibSp', 'Parch', 'Families'], axis=1)
data_test = data_test.drop(['SibSp', 'Parch', 'Families'], axis=1)
data_integrate = [data_train, data_test]

# 创建Age*Pclass特征
for d in data_integrate:
    d['Age*Pclass'] = d.Age * d.Pclass

# 去除Age与Pclass特征
# data_train = data_train.drop(['Age', 'Pclass'], axis=1)
# data_test = data_test.drop(['Age', 'Pclass'], axis=1)
# data_integrate = [data_train, data_test]

# 将Embarked的缺失值填充为众数
for d in data_integrate:
    d.Embarked = d.Embarked.fillna(value=d.Embarked.dropna().mode()[0])

# 将Embarked数据类型转化为数值型
for d in data_integrate:
    d.Embarked = d.Embarked.map({'S': 0, 'C': 1, 'Q': 2})

# 填充Fare的缺失值为中位数
for d in data_integrate:
    d.Fare = d.Fare.fillna(d.Fare.dropna().median())

# 将Fare划分为4个区间
data_train['Fare_group'] = pd.qcut(data_train.Fare, 4)

for d in data_integrate:
    d.loc[(d.Fare < 7.91), 'Fare'] = 0
    d.loc[(d.Fare >= 7.91) & (d.Fare < 14.454), 'Fare'] = 1
    d.loc[(d.Fare >= 14.454) & (d.Fare < 31.0), 'Fare'] = 2
    d.loc[(d.Fare >= 31.0), 'Fare'] = 3

data_train = data_train.drop('Fare_group', axis=1)

# 将目标变量存活率移动到最后一列
data_target = data_train.pop('Survived')
data_train = pd.concat([data_train, data_target], axis=1)
# print(data_train)
# print(data_test)


# 计算基尼系数（二分类）
def cal_gini(data, colIndex, split):
    count = 0
    right_dict = {}
    left_dict = {}
    for rI in range(len(data)):
        row = data.iloc[rI]
        if row[colIndex] >= split:
            count += 1
            if row[-1] not in right_dict:
                right_dict[row[-1]] = 0
            right_dict[row[-1]] += 1
        else:
            if row[-1] not in left_dict:
                left_dict[row[-1]] = 0
            left_dict[row[-1]] += 1
    right_data_ratio = count / len(data)
    left_data_ratio = 1 - right_data_ratio
    gini = [0] * 2
    gini[0] = right_data_ratio * (1 - sum([math.pow(right_dict[x] / count, 2) for x in right_dict]))
    gini[1] = left_data_ratio * (1 - sum([math.pow(left_dict[x] / (len(data) - count), 2) for x in left_dict]))
    total_gini = sum(gini)
    return total_gini


# 选择最小基尼系数进行数据切分（二分法）
def dataSplit(data):
    feature_num = len(data.iloc[0]) - 1  # 数据的特征总数
    # print(feature_num)
    mini_gini = float("inf")
    splitPoint = None
    gini_ls = []
    feature_ls = []  # 为保存最优划分属性预设列表
    splitPt_ls = []  # 为保存切分点预设列表

    for fe in range(feature_num):
        data_col = data.iloc[:, fe]
        data_col_set = set(data_col)  # 该特征下的取值集合（去重）

        # 特征取值只有两个时，选取两值的平均值作为切分点
        if len(data_col_set) <= 2:
            splitPt = sum(data_col_set)/2   # 取中间值
            # print(splitPoint)
            gini = cal_gini(data, fe, splitPt)
            gini_ls.append(gini)
            feature_ls.append(fe)
            splitPt_ls.append(splitPt)

        else:
            data_col_list = list(data_col_set)
            data_col_list.sort()
            split_num = len(data_col_list) - 1
            for i in range(split_num):
                splitPt = (data_col_list[i] + data_col_list[i + 1]) / 2
                gini = cal_gini(data, fe, splitPt)
                # print(gini)
                gini_ls.append(gini)
                feature_ls.append(fe)
                splitPt_ls.append(splitPt)

    mini_gini = min(gini_ls)  # 选择最小的基尼系数
    mini_gini_index = gini_ls.index(mini_gini)
    splitColIndex = feature_ls[mini_gini_index]
    splitFeature = data.columns[splitColIndex]  # 最优划分属性
    splitPoint = splitPt_ls[mini_gini_index]  # 切分点
    splitCol = data.iloc[:, splitColIndex]  # 用以划分数据的列
    sample_num = splitCol.shape[0]  # 样本总数

    right_ls = []
    left_ls = []
    for i in range(sample_num):
        if splitCol[i] >= splitPoint:
            right_ls.append(i)
        else:
            left_ls.append(i)
    del_data = data.pop(data.columns[splitColIndex])  # 去掉已经用于划分的特征列
    # 将提取数据先转化为array类型以重置行索引，方便后续数据划分
    dataColLable = data.columns
    left_array = np.array(data.iloc[left_ls, :])
    right_array = np.array(data.iloc[right_ls, :])
    left_data = pd.DataFrame(left_array)
    right_data = pd.DataFrame(right_array)
    # 由array转换为DataFrame类型时，列标签默认为整型，所以需要修改列标签
    left_data.columns = dataColLable
    right_data.columns = dataColLable
    return splitFeature, splitPoint, left_data, right_data


# 生成决策树
def growTree(train_data):
    label_num = len(set(train_data.iloc[:, -1]))
    # 数据集为空
    if label_num == 0:
        return
    # 剩余样本都为同一类别，则返回该类别
    if label_num == 1:
        label = train_data.iloc[0, -1]
        return label
    # 剩余样本的特征数为0，则返回样本中出现次数最多的类别
    feature_num = len(train_data.iloc[0]) - 1
    if feature_num == 0:
        # 返回最后一列类别的众数，即返回出现次数最多的类别
        label = list(train_data.iloc[:, -1].mode())  # 先转换成列表去掉行索引
        # print(label[0])
        return label[0]

    splitFea, splitPt, leftData, rightData = dataSplit(train_data)
    myTree = {splitFea: {}}  # 用字典保存生成的节点
    left = 0    # 属性取值小于切分点值
    right = 1   # 属性取值大于或等于切分点值
    myTree[splitFea][left] = growTree(leftData)
    myTree[splitFea][right] = growTree(rightData)
    myTree[splitFea]['pt'] = splitPt    # 保存切分点，便于后续测试时划分数据集
    # print(myTree)
    return myTree


# 决策树的分类函数
def classify(mytree, testList, columns):
    first_str = list(mytree.keys())[0]  # 分支点
    second_dict = mytree[first_str]
    col_index = columns.index(first_str)  # 列标签的索引
    classLabel = None  # 预设类别标签
    if testList[col_index] < second_dict['pt']:
        if type(second_dict[0]).__name__ == "dict":
            classLabel = classify(second_dict[0], testList, columns)
        else:
            classLabel = second_dict[0]
    else:
        if type(second_dict[1]).__name__ == "dict":
            classLabel = classify(second_dict[1], testList, columns)
        else:
            classLabel = second_dict[1]

    return classLabel


def cartTree(train_data, test_data):
    my_tree = growTree(deepcopy(train_data))  # 使用深拷贝，因为后面需要用训练集进行测试
    # print(my_tree)
    columns = list(test_data.columns)    # 列标签
    # print(columns)
    test_length = test_data.shape[0]
    predict_label = [0]*test_length  # 为存放预测结果预设列表
    accurate_sum = 0
    for row_index in range(test_length):
        test_ls = list(test_data.iloc[row_index])
        predict_label[row_index] = (classify(my_tree, test_ls, columns))
        if test_ls[-1] == predict_label[row_index]:
            accurate_sum += 1
    accurate_rate = accurate_sum / test_length  # 精确度

    return predict_label, accurate_rate
    # return predict_label


result, accuracy = cartTree(data_train, data_train)
# result = cartTree(data_train, data_test)
print(result)
print("精确度为{:%}".format(accuracy))

# data_reserve = pd.read_csv('test.csv')
# data_reserve['predict'] = result
# data_reserve.to_csv('test_predict.csv', sep=',', index=False)

data_reserve = pd.read_csv('train.csv')
data_reserve['predict'] = result
data_reserve.to_csv('train_predict.csv', sep=',', index=False)