import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["font.family"] = "SimHei"
mpl.rcParams["axes.unicode_minus"] = False  # 显示负号


class LogisticRegression:
    '''numpy实现逻辑回归算法'''

    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def sigmoid(self, z):
        # return 1>=value>=0.5, 类别为1
        # return 0<=value<0.5，类别为0
        return 1.0/(1.0 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        # 初始化权重和截距
        self.w_ = np.zeros(1 + X.shape[1])

        # 损失列表
        self.loss_ = []

        # 迭代
        for i in range(self.epochs):
            z = np.dot(X, self.w_[1:]) + self.w_[0]

            # 计算概率值（判定为1的概率值）
            prob = self.sigmoid(z)

            # 根据逻辑回归的损失函数，计算损失值
            loss = -np.sum(y*np.log(prob) + (1-y)*np.log(1-prob))
            self.loss_.append(loss)

            # 根据逻辑回归的梯度函数，更新权重
            self.w_[1:] += self.learning_rate * np.dot(X.T, (y-prob))
            self.w_[0] += self.learning_rate * np.sum(y-prob)

    def predict_prob(self, X):
        '''预测样本所属类别的概率'''
        X = np.asarray(X)
        z = np.dot(X, self.w_[1:]) + self.w_[0]
        prob = self.sigmoid(z)
        prob = prob.reshape(-1, 1)
        return np.concatenate([1-prob, prob], axis=1)  # 横向拼接

    def predict(self, X):
        '''预测样本所属类别'''
        return np.argmax(self.predict_prob(X), axis=1)


def print_plot(y_test, y_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, 'ro', ms=15, label="实际值")  # ms: 指定圆大小
    plt.plot(y_pred, 'go', label="预测值")
    plt.title("逻辑回归\n预测鸢尾花种类")
    plt.xlabel("")
    plt.ylabel("")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 加载数据
    df_data = pd.read_csv("./dataset/iris.arff.csv", header=0)
    # print(df_data.head(3))
    # print(df_data.info())
    # print(df_data.describe())
    # print(df_data.duplicated().any())
    # print(set(df_data["class"]))  # 查看包含几个类别

    # 删除重复数据
    # if df_data.duplicated().any():
    # df_data.drop_duplicates(inplace=True)

    # 类别映射为数字
    df_data["class"] = df_data["class"].map(
        {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})

    # 逻辑回归，预测二分类，鸢尾花数据存在3种类别，只取2个类别数据
    df_data_0 = df_data[df_data["class"] == 0]
    df_data_1 = df_data[df_data["class"] == 1]

    # 打乱数据
    df_data_0 = df_data_0.sample(len(df_data_0), random_state=0)
    df_data_1 = df_data_1.sample(len(df_data_1), random_state=0)

    # 切分数据（每种数据约50条记录）
    X_train = pd.concat(
        [df_data_0.iloc[:40, :-1], df_data_1.iloc[:40, :-1]], axis=0)
    y_train = pd.concat(
        [df_data_0.iloc[:40, -1], df_data_1.iloc[:40, -1]], axis=0)
    X_test = pd.concat(
        [df_data_0.iloc[40:, :-1], df_data_1.iloc[40:, :-1]], axis=0)
    y_test = pd.concat(
        [df_data_0.iloc[40:, -1], df_data_1.iloc[40:, -1]], axis=0)

    # 鸢尾花的特征都在同一数量级，可以不用进行标准化
    # 训练模型
    logisticR = LogisticRegression()
    logisticR.fit(X_train, y_train)

    # 预测，评估
    y_pred = logisticR.predict(X_test)
    print("accuracy: ", np.sum(y_pred == y_test)/len(y_test))

    # 绘制预测结果
    print_plot(y_test.values, y_pred)
