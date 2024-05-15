import matplotlib.pyplot as plt
import numpy as np

# 创建x和y坐标的列表

x1 = [1, 1.3, 1.5, 1.2, 1.4, 1.3, 3.1, 3.3, 3.4, 3.2, 3.5, 3.4]

x2 = [1, 1.4, 1.2, 1.3, 1.5, 1.4, 3.1, 3.4, 3.2, 3.3, 3.2, 3.5]

min_x1 = x1[0]
min_var = 100
for i in range(1, len(x1) - 1):
    n_x1_l = np.array(x1[:i])
    n_x1_r = np.array(x1[i:])
    if (min_var > np.var(n_x1_l) * i + np.var(n_x1_r) * (len(x1) - i)):
        min_var = np.var(n_x1_l) * i + np.var(n_x1_r) * (len(x1) - i)
        min_x1 = i

print(min_var, min_x1)
print(x1[:min_x1], x1[min_x1:])

min_x2 = x2[0]
min_var2 = 100
for i in range(1, len(x2) - 1):
    n_x2_l = np.array(x2[:i])
    n_x2_r = np.array(x2[i:])
    if (min_var2 > np.var(n_x2_l) * i + np.var(n_x2_r) * (len(x2) - i)):
        min_var2 = np.var(n_x2_l) * i + np.var(n_x2_r) * (len(x2) - i)
        min_x2 = i

print(min_var2, min_x2)
print(x2[:min_x2], x2[min_x2:])
# 绘制点图

# plt.scatter(x1, x2)
#
# # 添加标题和坐标轴标签
#
# plt.title("Point Chart")
#
# plt.xlabel("x1")
#
# plt.ylabel("x2")
#
# plt.show()
