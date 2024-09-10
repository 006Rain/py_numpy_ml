import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["font.family"] = "SimHei"
mpl.rcParams["axes.unicode_minus"] = False  # 显示负号
PBAR_ASCII = '.>>>>>>>>>='


class KMeans():
    '''numpy实现KMeans算法'''

    def __init__(self, k=3, epochs=100):
        self.k = k  # 簇个数
        self.epochs = epochs

    def fit(self, X):
        X = np.array(X)
        np.random.seed(0)

        # 从数据集中随机选择K个点作为初始聚类中心
        self.cluster_centers_ = X[np.random.randint(0, len(X), self.k)]

        # 存放数据类别标签
        self.labels_ = np.zeros(len(X))

    # 在这里添加你的代码

        # 迭代
        for t in tqdm(range(self.epochs), desc="training...", ascii=PBAR_ASCII):
            # 遍历样本
            for index, x in enumerate(X):
                # 计算样本与聚类中心的欧式距离
                distance = np.sqrt(
                    np.sum((x - self.cluster_centers_)**2, axis=1))

                # 将最小距离的索引赋值给标签数组
                # 索引的值就是当前所属的簇，范围（0，k-1）
                self.labels_[index] = distance.argmin()

            # 更新聚类中心
            for i in range(self.k):
                # 计算每个簇内所有点的均值，用于更新聚类中心
                self.cluster_centers_[i] = np.mean(
                    X[self.labels_ == i], axis=0)

    def predict(self, X):
        X = np.asarray(X)
        result = np.zeros(len(X))

        for index, x in enumerate(X):
            # 计算样本与聚类中心的距离
            distance = np.sqrt(np.sum((x - self.cluster_centers_)**2, axis=1))

            # 找到距离聚类中心最近的一个类别
            result[index] = distance.argmin()

        return result


def print_plot(df_data, cluster_centers, labels):
    plt.figure(figsize=(10, 5))

    # 绘制每个类别的散点图，取前2列数据
    plt.scatter(df_data[labels == 0].iloc[:, 0],
                df_data[labels == 0].iloc[:, 1], label="类别1")
    plt.scatter(df_data[labels == 1].iloc[:, 0],
                df_data[labels == 1].iloc[:, 1], label="类别2")
    plt.scatter(df_data[labels == 2].iloc[:, 0],
                df_data[labels == 2].iloc[:, 1], label="类别3")

    # 绘制聚类中心
    plt.scatter(cluster_centers[:, 0],
                cluster_centers[:, 1], marker="*", c='yellow', s=100)

    plt.title("KMeans\n食物与肉类购买的聚类分析")
    plt.xlabel("食物")
    plt.ylabel("肉类")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 加载数据
    df_data = pd.read_csv("./dataset/order.csv", header=0)
    # print(df_data.head(3))
    # print(df_data.duplicated().any())
    # print(df_data.columns)

    # 只取购买的物品列数据作为数据集
    df_data = df_data.iloc[:, -8:]
    # print(df_data.head())

    # 训练模型
    kmeans = KMeans(3, 50)
    kmeans.fit(df_data)
    # print(kmeans.cluster_centers_)
    # print(kmeans.labels_)
    # print(df_data[kmeans.labels_ == 0].head())

    # 预测
    y_pred = kmeans.predict([[30, 30, 40, 0, 0, 0, 0, 0], [
                            0, 0, 0, 0, 0, 30, 30, 40], [30, 30, 0, 0, 0, 0, 20, 20]])
    print("y_pred: ", y_pred)

    # 绘制散点图
    print_plot(df_data, kmeans.cluster_centers_, kmeans.labels_)
