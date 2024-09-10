import pandas as pd
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt

mlp.rcParams["font.family"] = "SimHei"
mlp.rcParams["axes.unicode_minus"] = False  # 显示负号


class StandardScaler:
    '''对数据集进行标准化'''
    '''将列数据处理成均值为0，标准差为1的正太分布数据'''

    def fit(self, X):
        X = np.asarray(X)
        self.mean_ = np.mean(X, axis=0)  # 按列计算均值
        self.std_ = np.std(X, axis=0)  # 按列计算标准差

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class LinearRegression():
    '''梯度下降法实现线性回归'''

    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        # 初始化权重，w0为偏置
        self.w_ = np.zeros(1+X.shape[1])

        # 损失列表
        # 损失公式：1/2.0 * (y - y_pred)^2
        self.loss_ = []

        # 迭代，使用梯度下降法，更新权重和偏置
        for i in range(self.epochs):
            y_hat = np.dot(X, self.w_[1:]) + self.w_[0]

            # 计算误差
            error = y_hat - y  # y_hat - y: 更新梯度的负方向
            # error = y - y_hat # y - y_hat: 更新梯度的正方向
            if (len(error[error > 10000]) > 0):
                print("error: ", error[error > 10000][:3], y[:3], y_hat[:3])

            # 计算损失函数值
            self.loss_.append(np.sum(error**2/2.0))

            # 根据误差，调整权重和偏置
            self.w_[1:] += -self.learning_rate * np.dot(X.T, error)
            self.w_[0] += -self.learning_rate * np.sum(error)

    def predict(self, X):
        X = np.asarray(X)
        result = np.dot(X, self.w_[1:]) + self.w_[0]
        return result


def print_plot(y_test, y_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, 'ro-', label='实际房价')
    plt.plot(y_pred, 'go-', label='预测房价')
    plt.title("梯度下降法线性回归\n波士顿房价预测")
    plt.xlabel("序号")
    plt.ylabel("房价")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 加载数据
    df_data = pd.read_csv("./dataset/boston.csv", header=0)
    # print(df_data.head())
    # print(df_data.describe())
    df_data = df_data.iloc[:, 1:]  # 取消无用列：序号列
    # print(df_data.duplicated().any()) # 查看是否有重复数据

    # 打乱数据
    df_data = df_data.sample(len(df_data), random_state=0)

    # 切分数据(数据集共506条数据)
    X_train = df_data.iloc[:400, :-1]
    y_train = df_data.iloc[:400, -1]
    X_test = df_data.iloc[400:, :-1]
    y_test = df_data.iloc[400:, -1]

    # 数据标准化
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.fit_transform(X_test)
    ss2 = StandardScaler()
    y_train = ss2.fit_transform(y_train)
    y_test = ss2.fit_transform(y_test)

    # 训练模型
    linearR = LinearRegression(0.0005, 20)
    linearR.fit(X_train, y_train)

    # 预测
    y_pred = linearR.predict(X_test)

    # 评估
    print("rmse: ", np.sqrt(np.mean((y_pred - y_test) ** 2)))

    #
    print_plot(y_test.values, y_pred)
