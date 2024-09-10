import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["font.family"] = "SimHei"
mpl.rcParams["axes.unicode_minus"] = False  # 显示负号


class LinearRegression:
    '''最小二乘法实现线性回归'''

    def fit(self, X, y):
        # 将X，y转换为矩阵
        # 因为后面需要进行矩阵运算
        X = np.asmatrix(X)
        y = np.asmatrix(y)
        y = y.reshape(-1, 1)  # 矩阵应为二维

        # 最小二乘法公式
        self.w_ = (X.T * X).I * X.T * y

    def predict(self, X):
        # X.copy是防止传入的是数组的部分数据的索引
        X = np.asmatrix(X.copy())
        result = X * self.w_

        # ravel: 将多维数据展平，变为一维数据
        return np.array(result).ravel()


def print_plot(y_test, y_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, 'ro-', label="实际价格")
    plt.plot(y_pred, 'go-', label="预测价格")
    plt.title("最小二乘法线性回归\n波士顿房价预测")
    plt.xlabel("序号")
    plt.ylabel("房价")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 加载波士顿房价数据
    df_data = pd.read_csv("./dataset/boston.csv")
    # print(df_data.head())
    # print(df_data.info()) # 查看数据是否存在缺失值
    df_data = df_data.iloc[:, 1:]  # 去掉无用数据：序号列
    # print(df_data.head())
    df_data.insert(0, "intercept", value=1)  # 增加截距列：b
    # print(df_data.head())

    # 打乱数据
    df_data = df_data.sample(len(df_data), random_state=0)

    # 切分数据（数据共506条）
    X_train = df_data.iloc[:400, :-1]
    y_train = df_data.iloc[:400, -1]
    X_test = df_data.iloc[400:, :-1]
    y_test = df_data.iloc[400:, -1]

    linearR = LinearRegression()
    linearR.fit(X_train, y_train)
    y_pred = linearR.predict(X_test)
    print("mse: ", np.mean((y_pred - y_test.values)**2))
    print("mae: ", np.mean(np.abs(y_pred - y_test.values)))
    print("bias: ", linearR.w_[0])
    print("weights: ", linearR.w_[1:])
    print_plot(y_test.values, y_pred)

    # 波士顿房价数据集介绍：
    # crim: 房屋所在镇犯罪率
    # zn: 面积大于25000平方英尺住宅所占比例
    # indus: 房屋所在镇非零售区域所占比例
    # chas: 房屋是否位于河边，河边为1，否则0
    # nox: 一氧化氮浓度
    # rm: 平均房间数量
    # age: 1940年前建成房屋所占的比例
    # dis: 房屋距离波士顿五大就业中心的加权距离
    # rad: 辐射性公路的接近指数
    # tax: 每10000美元的全值财产税率
    # ptratio: 房屋所在镇师生比例
    # black: 房屋所在镇黑人比例
    # lstat: 弱势人口占比
    # medv: 房屋价格中位数（单位：千美元）
