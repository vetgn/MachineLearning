from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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

# 定义四种核函数
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

# 训练并打印准确率
for kernel in kernels:
    classifier = svm.SVC(kernel=kernel, gamma='auto', decision_function_shape='ovo')
    classifier.fit(train_data, train_label.ravel())

    train_accuracy = classifier.score(train_data, train_label)
    test_accuracy = classifier.score(test_data, test_label)

    print(f"Kernel: {kernel}, 训练集准确率: {train_accuracy}, 测试集准确率: {test_accuracy}")
