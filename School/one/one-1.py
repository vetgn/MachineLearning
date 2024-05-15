# 使用基于类目特征的朴素贝叶斯
# 模拟数据，测试可知，CategoricalNB可以接受数字的字符串，但是不可以出现英文和中文

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
import numpy as np

df = pd.read_csv("../datas/knn3.txt", delimiter="\t")  # 读取数据，df类型

# 对文本型特征进行标签编码
label_encoder = LabelEncoder()
categorical_columns = ['空气', '云', '阳光', '天气分类']
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])  # 按标签重新转换列，并覆盖原来的列

# 读取相应的列
X = df[['湿度', '温度', '空气', '云', '阳光']]
y = df['天气分类']
# 转换成array
X_train = np.array(X).reshape(16, -1)
y_train = np.array(y).T
clf = CategoricalNB()
clf.fit(X_train, y_train)
# 选择X_train进行测试
new_pred = np.array(X_train[0]).reshape(1, -1)
print(clf.predict_proba(new_pred)) # [[0.95219456 0.04780544]]
