import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# 这里我们创建了50个数据点，并将它们分为了2类
x, y = make_blobs(n_samples=50, centers=2, random_state=6)
# make_blobs函数是为聚类产生数据集，产生一个数据集和相应的标签;centers:表示类别数（标签的种类数），默认值3
# random_state:官网解释是随机生成器的种子，可以固定生成的数据，给定数之后，每次生成的数据集就是固定的。
print(y)
# 构建一个内核为线性的支持向量机模型
clf = svm.SVC(kernel="linear", C=1000)
clf.fit(x, y)
plt.scatter(x[:, 0], x[:, 1], c=y, s=30, cmap='Paired')
# 建立图形坐标
ax = plt.gca()  # gca:get current ax:获取当前坐标
xlim = ax.get_xlim()  # 获取数据点x坐标的最大值和最小值
ylim = ax.get_ylim()  # 获取数据点y坐标的最大值和最小值

# 根据坐标轴生成等差数列(这里是对参数进行网格搜索)
# 在坐标轴范围内生成30个均匀分布的点
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)  # 创建网格坐标。
# 例如：xx:[0,1,2];yy:[0,1,2];XX=[[0,0,0],[1,1,1],[2,2,2]],YY=[[0,1,2],[0,1,2]，[0,1,2]]
# XX与YY分别记录一个正方形的网格的每个点的横坐标和纵坐标。
xy = np.vstack([XX.ravel(), YY.ravel()]).T  # 将网格坐标转换为一维数组，并组合成坐标对
# xy将XX与YY配对组成真正的坐标
Z = clf.decision_function(xy).reshape(XX.shape)  # 先执行决策函数，输出样本点到分界线的距离。然后转成网格的样式，对应着XX、YY的取值。 计算每个网格点到决策边界的距离，并重塑为与网格相同形状的数组。

# 画出分类的边界
ax.contour(XX, YY, Z, cmap='brg', levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"])
# 在平面上，根据Z的值，绘制出Z=-1；Z=0；Z=1的等高线。
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidths=1, facecolors="none")
# 绘制第1和第2个支持向量点
plt.show()
print("支持向量的点:")
print(clf.support_vectors_)
