from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 计算指标
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("mse: ", mse)
print("r2: ", r2)
