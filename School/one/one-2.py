import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB  # 导入高斯朴素贝叶斯
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("../datas/knn3.txt", delimiter="\t")  # 读取数据，df类型

# 对文本型特征进行标签编码
label_encoder = LabelEncoder()
categorical_columns = ['空气', '云', '阳光', '天气分类']
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])  # 按标签重新转换列，并覆盖原来的列

# 转换成array
X = np.array(df[['湿度', '温度', '空气', '云', '阳光']]).reshape(16, -1)
y = np.array(df['天气分类']).T
# 划分训练集和测试集，测试集比例16%
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.16, random_state=1)
# 训练贝叶斯分类模型
model = GaussianNB()
model.fit(train_X, train_y)
expected = test_y  # 实际类别值[0, 1, 0]
predicted = model.predict(test_X)  # 预测的类别值[0, 0, 0]
print(metrics.classification_report(expected, predicted, labels=[0, 1], zero_division=1))  # 输出分类信息
label = list(set(y))  # 去重复，得到标签类别
print(metrics.confusion_matrix(expected, predicted, labels=label))  # 输出混淆矩阵
#               precision    recall  f1-score   support
#
#            0       0.67      1.00      0.80         2
#            1       1.00      0.00      0.00         1
#
#     accuracy                           0.67         3
#    macro avg       0.83      0.50      0.40         3
# weighted avg       0.78      0.67      0.53         3
#
# [[2 0]
#  [1 0]]
