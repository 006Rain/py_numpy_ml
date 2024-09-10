import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["font.family"] = "SimHei"
mpl.rcParams["axes.unicode_minus"] = False  # 显示负号


class KNN:
    '''KNN回归算法'''

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y)

    def predict(self, X):
        X = np.asarray(X)
        result = []

        for x in X:
            distance = np.sqrt(np.sum((x - self.X) ** 2, axis=1))
            sorted_index = np.argsort(distance)
            top_k_index = sorted_index[:self.k]
            top_k_y = self.y[top_k_index]
            # 相邻k个点的值的均值作为预测结果
            y_pred = np.mean(top_k_y)
            result.append(y_pred)

        return np.asarray(result)

    def predict_with_weights(self, X):
        X = np.asarray(X)
        result = []

        for x in X:
            distance = np.sqrt(np.sum((x - self.X) ** 2, axis=1))
            sorted_index = np.argsort(distance)
            top_k_index = sorted_index[:self.k]
            top_k_y = self.y[top_k_index]

            # 前k个点各自的权重：
            # 计算前k个点的倒数和aa
            # 计算前k个点的倒数BB
            # weights = BB / aa
            # 使用权重，计算预测结果
            aa = np.sum(1.0/(distance[top_k_index] + 1e-4))
            BB = 1.0 / (distance[top_k_index] + 1e-4)
            weights = BB/aa
            y_pred = np.sum(top_k_y * weights)
            result.append(y_pred)

        return np.asarray(result)


def plot_result(y_test, y_pred, y_pred2):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, "ro-", label="实际值")  # 红色，实线
    plt.plot(y_pred, "bo--", label="预测值")  # 蓝色，虚线
    plt.plot(y_pred2, "go--", label="权重预测值")  # 绿色，虚线
    plt.title("KNN连续值预测")
    plt.xlabel("序号")
    plt.ylabel("花瓣长度")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 加载数据
    df_data = pd.read_csv("./dataset/iris.arff.csv")
    df_data.drop_duplicates(inplace=True)
    df_data = df_data.sample(len(df_data), random_state=0)
    print(df_data.head())

    # 使用前3个特征，预测第4个特征的值
    # 故，去除类别列数据
    df_data.drop("class", axis=1, inplace=True)
    print(df_data.head())

    # 切分数据集（数据大约150条记录）.iloc
    X_train = df_data.iloc[:120, :-1]
    y_train = df_data.iloc[:120, -1]
    X_test = df_data.iloc[120:, :-1]
    y_test = df_data.iloc[120:, -1]

    # 训练模型
    knn = KNN(3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # 评估
    mae = np.mean(np.abs(y_test - y_pred))
    print("mae: ", mae)
    print("y_test - y_pred: ", y_test.values - y_pred)

    print("**************************")

    # 考虑权重，预测+评估
    y_pred2 = knn.predict_with_weights(X_test)

    mae2 = np.mean(np.abs(y_test - y_pred2))
    print("mae2: ", mae2)
    print("y_test - y_pred2: ", y_test.values - y_pred2)

    # 展示预测结果
    plot_result(y_test.values, y_pred, y_pred2)
