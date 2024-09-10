import pandas as pd
import numpy as np


def load_data(data_file_path):
    # head=0: 第0行为标题
    df_iris = pd.read_csv(data_file_path, header=0)
    # print("df_iris: \n", df_iris.head(3))
    # print("df_iris: \n", df_iris.sample(10))

    # 删除重复记录
    if (df_iris.duplicated().any()):
        df_iris.drop_duplicates(inplace=True)

    # 查看各个类别的个数
    # print("class count: ", df_iris["class"].value_counts())

    # 将类别名映射为数字
    df_iris["class"] = df_iris["class"].map(
        {"Iris-setosa": 0, "Iris-virginica": 1, "Iris-versicolor": 2})

    return df_iris


# numpy实现KNN分类算法
class KNN:
    def __init__(self, k=3):
        # 参考的最近距离点数目
        self.k = k

    def fit(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y)

    def predict(self, X):
        X = np.asarray(X)
        result = []  # 返回的预测值列表

        # 对每条记录进行预测
        for x in X:
            # 计算待预测数据特征与训练集中每条记录的特征的欧式距离
            # x - self.X为2维数据，np.sum取axis=1，既对第2个维度求和
            distance = np.sqrt(np.sum((x - self.X)**2, axis=1))
            # 对欧式距离列表进行排序，返回下标
            sorted_index = np.argsort(distance)
            # 取前k个下标
            top_k_index = sorted_index[:self.k]
            # 取下标对应的标签值，统计每个标签值出现的次数
            y_value_count = np.bincount(self.y[top_k_index])
            # 因为将鸢尾花类型值映射为了0,1,2
            # 故：取个数最多的标签对应的下标既可作为预测结果
            y_pred = y_value_count.argmax()
            result.append(y_pred)

        return np.asarray(result)

    # 考虑权重：距离近的权重大，距离远的权重小
    def predict_with_weights(self, X):
        X = np.asarray(X)
        result = []

        for x in X:
            distance = np.sqrt(np.sum((x - self.X)**2, axis=1))
            sorted_index = np.argsort(distance)
            top_k_index = sorted_index[:self.k]
            top_k_y = self.y[top_k_index]

            # 若不指定bincount参数weights，则统计时对出现的统计值+1
            # 若指定，则对出现的统计值+weight
            y_value_count = np.bincount(
                top_k_y, weights=1.0/distance[top_k_index])
            y_predict = y_value_count.argmax()
            result.append(y_predict)

        return result


if __name__ == '__main__':

    # 加载鸢尾花数据
    df_iris = load_data("./dataset/iris.arff.csv")
    # print(df_iris.head())

    # 数据存在3各类别，随机取出训练和测试数据，可能导致某类别数据训练集中存在过少
    # 从而导致训练模型无法准确预测该类别，故分别取出各类别数据，然后均匀切分数据集
    df_class_0 = df_iris[df_iris["class"] == 0]
    df_class_1 = df_iris[df_iris["class"] == 1]
    df_class_2 = df_iris[df_iris["class"] == 2]
    # print(df_class_0.head())

    # 打乱数据顺序（使用全采样，得到随机顺序数据）
    df_class_0 = df_class_0.sample(len(df_class_0), random_state=0)
    df_class_1 = df_class_1.sample(len(df_class_1), random_state=0)
    df_class_2 = df_class_2.sample(len(df_class_2), random_state=0)
    # print(df_class_0.head())

    # 切分训练集和测试集（加载数据时，查看到每个类别数据约为50条）
    df_X_train = pd.concat([df_class_0.iloc[:40, :-1],
                            df_class_1.iloc[:40, :-1], df_class_2.iloc[:40, :-1]])
    df_y_train = pd.concat([df_class_0.iloc[:40, -1],
                            df_class_1.iloc[:40, -1], df_class_2.iloc[:40, -1]])

    df_X_test = pd.concat([df_class_0.iloc[40:, :-1],
                           df_class_1.iloc[40:, :-1], df_class_2.iloc[40:, :-1]])
    df_y_test = pd.concat(
        [df_class_0.iloc[40:, -1], df_class_1.iloc[40:, -1], df_class_2.iloc[40:, -1]])

    # print(df_X_train.head())
    # print(df_y_train.head())

    # 训练、预测
    knn = KNN(3)
    knn.fit(df_X_train, df_y_train)
    y_pred = knn.predict(df_X_test)

    # 评估
    print("predict success: ", np.sum((df_y_test == y_pred)))
    print("predict faile: ", np.sum((df_y_test != y_pred)))
    print("predict accurate: ", np.sum(df_y_test == y_pred)/len(df_y_test))

    print("*************")

    # 带权重的预测与评估
    y_pred2 = knn.predict_with_weights(df_X_test)

    print("predict success: ", np.sum((df_y_test == y_pred)))
    print("predict faile: ", np.sum((df_y_test != y_pred)))
    print("predict accurate: ", np.sum(df_y_test == y_pred)/len(df_y_test))
