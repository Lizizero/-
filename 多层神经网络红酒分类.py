import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# 读取数据
data = pd.read_csv('wine.csv')

# 特征缩放
scaler = StandardScaler()
scaler.fit(data.iloc[:, :-1])
X_scaled = scaler.transform(data.iloc[:, :-1])

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X_scaled, data.iloc[:, -1], test_size=0.2, random_state=0)

# 搭建多层神经网络
mlp = MLPClassifier(solver='lbfgs',
                    activation='logistic',
                    hidden_layer_sizes=(15, 10),
                    max_iter=1000,
                    random_state=0)

mlp.fit(X_train, y_train)

# 模型评估
print("Accuracy on training set: {:.2f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score(X_test, y_test)))
