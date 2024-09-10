import pandas as pd
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt

mlp.rcParams["font.family"] = "SimHei"
mlp.rcParams["axes.unicode_minus"] = False  # 显示负号


class Perception:
    '''numpy实现感知器算法'''

    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def step(self, z):
        '''阶跃函数
          z>=0: return 1
          z<0: return -1
        '''
        return np.where(z > 0, 1, -1)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        # 初始化权重和偏置
        self.w_ = np.zeros(1+X.shape[1])

        # 损失记录列表
        self.loss_ = []

        # 迭代
        # 感知器与逻辑回归的区别：
        # 逻辑回归中，使用所有样本计算梯度来更新权重
        # 感知器使用单个样本，依次计算梯度更新权重
        for i in range(self.epochs):
            loss = 0
            for x, target in zip(X, y):
                # 计算预测值
                y_hat = self.step(np.dot(x, self.w_[1:]) + self.w_[0])

                # 若预测值不等于目标值，loss+1，否则loss+0
                loss += y_hat != target

                # 更新权重
                self.w_[1:] += self.learning_rate * (target - y_hat) * x
                self.w_[0] += self.learning_rate * (target - y_hat)

            self.loss_.append(loss)

    def predict(self, X):
        return self.step(np.dot(X, self.w_[1:]) + self.w_[0])


def print_plot(y_test, y_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, 'ro', ms=15, label="实际类别")  # ms: 圆的大小
    plt.plot(y_pred, 'go', label="预测类别")
    plt.title("感知器\n预测鸢尾花类别")
    plt.xlabel("序号")
    plt.ylabel("类别")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 加载数据
    df_data = pd.read_csv("./dataset/iris.arff.csv", header=0)
    df_data.drop_duplicates(inplace=True)
    df_data["class"] = df_data["class"].map(
        {"Iris-setosa": 0, "Iris-virginica": -1, "Iris-versicolor": 1})
    print("class counts: ", df_data["class"].value_counts())

    # 感知器预测二分类，故只取2类数据
    df_data_0 = df_data[df_data["class"] == 1]
    df_data_1 = df_data[df_data["class"] == -1]

    # 切分数据（每种数据约50条）
    X_train = pd.concat(
        [df_data_0.iloc[:40, :-1], df_data_1.iloc[:40, :-1]], axis=0)
    y_train = pd.concat(
        [df_data_0.iloc[:40, -1], df_data_1.iloc[:40, -1]], axis=0)
    X_test = pd.concat(
        [df_data_0.iloc[40:, :-1], df_data_1.iloc[40:, :-1]], axis=0)
    y_test = pd.concat(
        [df_data_0.iloc[40:, -1], df_data_1.iloc[40:, -1]], axis=0)

    # 训练模型
    perception = Perception()
    perception.fit(X_train, y_train)

    # 预测，评估
    y_pred = perception.predict(X_test)
    print("accurate: ", np.sum(y_pred == y_test)/len(y_test))

    # 打印图标
    print_plot(y_test.values, y_pred)
