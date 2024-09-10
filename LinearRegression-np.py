import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle


# 定义损失函数
def linear_loss(X, y, w, b):
    num_train = X.shape[0]
    num_feature = X.shape[1]
    y_hat = np.dot(X, w) + b
    loss = np.sum((y_hat - y)**2)/num_train

    # 参数w的一阶导数
    dw = np.dot(X.T, (y_hat - y)) / num_train

    # 参数b的一阶导数
    db = np.sum(y_hat - y) / num_train

    return y_hat, loss, dw, db


# 初始权重、偏置参数w，b
def init_params(dims):
    w = np.zeros((dims, 1))
    b = 0
    return w, b


# 定义模型训练流程
def linear_train(X, y, learning_rate=0.01, epochs=10000):
    # 记录训练损失
    loss_his = []

    # 初始化模型参数
    w, b = init_params(X.shape[1])

    # 迭代训练
    for i in range(1, epochs):
        # 计算损失
        y_hat, loss, dw, db = linear_loss(X, y, w, b)

        # 基于梯度下降法更新参数
        w += -learning_rate * dw
        b += -learning_rate * db

        # 保存每一次计算的损失
        loss_his.append(loss)

        # 每10000次，打印一次损失信息
        if i % 10000 == 0:
            print('epoch %d loss %f' % (i, loss))

        # 更新返回值
        params = {
            'w': w,
            'b': b
        }

        grads = {
            'dw': dw,
            'db': db
        }

    return loss_his, params, grads


# 使用训练得到的参数，预测
def predict(X, params):
    w = params['w']
    b = params['b']
    y_pred = np.dot(X, w) + b
    return y_pred


# 评估预测效果
def r2_score(y_test, y_pred):
    # 标签均值
    y_avg = np.mean(y_test)
    # 总离差平方和
    ss_total = np.sum((y_test - y_avg)**2)
    # 残差平方和
    ss_res = np.sum((y_test - y_pred)**2)
    # R2
    r2 = 1 - (ss_res / ss_total)

    return r2


if __name__ == '__main__':
    # 加载数据
    data_diabetes = load_diabetes()
    data, target = data_diabetes.data, data_diabetes.target
    X, y = shuffle(data, target, random_state=0)
    offset = int(0.8 * X.shape[0])
    X_train = X[:offset]
    X_test = X[offset:]
    y_train = y[:offset]
    y_test = y[offset:]

    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    # print("X_train shape: ", X_train.shape)
    # print("X_test shape: ", X_test.shape)
    # print("y_train shape: ", y_train.shape)
    # print("y_test shape: ", y_test.shape)
    # print("X_train 5: ", X_train[:5])
    # print("y_train 5: ", y_train[:5])

    # 训练模型
    loss_his, params, grads = linear_train(X_train, y_train, 0.01, 200000)
    print("params: ", params)

    # 预测
    y_pred = predict(X_test, params)
    r2 = r2_score(y_test, y_pred)
    print("r2: ", r2)
