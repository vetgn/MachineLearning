import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split


# 定义将标签转换为数字的函数
def Iris_label(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]


# 读取数据集
path = 'D:\Code\Folder-backup\MachineLearning\School\datas\iris\iris.data'
data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: Iris_label})

# 划分数据和标签
x, y = np.split(data, indices_or_sections=(4,), axis=1)
x = x[:, 0:2]
train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=0.6, test_size=0.4)

# 定义不同的gamma值
gammas = np.logspace(-10,1,50)

train_scores = []
test_scores = []

for gamma in gammas:
    classifier = svm.SVC(kernel='rbf', gamma=gamma, decision_function_shape='ovo')
    classifier.fit(train_data, train_label.ravel())

    train_score = classifier.score(train_data, train_label)
    test_score = classifier.score(test_data, test_label)

    train_scores.append(train_score)
    test_scores.append(test_score)

# 绘制学习曲线
plt.plot(gammas, test_scores)
plt.title("2107120101-朱军军", fontproperties='SimHei',fontsize='20')
plt.show()
