import csv
import numpy as np
from numpy.linalg import inv
import random
import sys
import math

data = []
# 每一维度存储一种污染物
for i in range(18):
    data.append([])

n_row = 0
text = open('data/train.csv','r',encoding='big5')
# print(text)
row  = csv.reader(text,delimiter=',')

for r in row:
    # 第0行是表头，不用读取
    if n_row != 0:
        # 每一行从第3列开始，27列结束，表示一天24小时的值
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0))
    n_row = n_row+1
text.close()

# x = []
# y = []
# # 每12个月
# for i in range(12):
#     # 1个月的前20天一共有480个小时，按滑动窗口取10小时，一共可以取471次
#     for j in range(471):
#         x.append([])
#         # 一共有18种污染物
#         for t in range(18):
#             # 连续9个小时
#             for s in range(9):
#                 x[471*i+j].append(data[t][480*i+j+s])
#         # x包含前9个小时18种污染物，y包含第10个小时的pm2.5
#         y.append(data[9][480*i+j+9])
# # print(type(x))
# x = np.array(x)
# # print(type(x))
# y = np.array(y)

# # 只去前9个小时的pm2.5作为feature
# x = []
# y = []
# # 每12个月
# for i in range(12):
#     # 1个月的前20天一共有480个小时，按滑动窗口取10小时，一共可以取471次
#     for j in range(471):
#         x.append([])
#         # 连续9个小时
#         for s in range(9):
#             #data的第10项是pm2.5，下标为9
#             x[471*i+j].append(data[9][480*i+j+s])
#         # x包含前9个小时18种污染物，y包含第10个小时的pm2.5
#         y.append(data[9][480*i+j+9])
# # print(type(x))
# x = np.array(x)
# # print(type(x))
# y = np.array(y)

# 只去前5个小时的pm2.5作为feature
x = []
y = []
# 每12个月
for i in range(12):
    # 1个月的前20天一共有480个小时，按滑动窗口取6小时，一共可以取476次
    for j in range(475):
        x.append([])
        # 连续5个小时
        for s in range(5):
            #data的第10项是pm2.5，下标为9
            x[475*i+j].append(data[9][480*i+j+s])
        # 归一化后发现结果更惨 ==
        # mean = sum(x[475*i+j])/5
        # temp= np.array(x[475*i+j])
        # temp = temp-mean
        # s_err = math.sqrt(np.dot(temp.transpose(),temp))
        # # 连续5个小时
        # for s in range(5):
        #     # data的第10项是pm2.5，下标为9
        #     if s_err!=0:
        #         # print('yes')
        #         x[475 * i + j][s] = (x[475 * i + j][s]-mean)/s_err
        # x包含前9个小时18种污染物，y包含第10个小时的pm2.5
        y.append(data[9][480*i+j+5])
# print(type(x))
x = np.array(x)
# print(type(x))
y = np.array(y)


# 加偏置值b，W0*1就是b，所以在x0前面插入1
x = np.concatenate((np.ones((x.shape[0],1)),x),axis=1)

# print(len(x[0])) # 18*9

# 初始化权重和其它超参数
w = np.zeros(len(x[0]))
l_rate = 10
repeat = 10000
# 正则项参数lamda
lamda = 0

x_t = x.transpose()
s_gra = np.zeros(len(x[0]))

for i in range(repeat):
    hypo = np.dot(x,w)
    loss = hypo-y
    cost = np.sum(loss**2)/len(x)
    cost_a = math.sqrt(cost)+lamda*(np.dot(w.transpose(),w))
    gra = np.dot(x_t,loss)
    gra = 2*(gra+lamda*w)
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w - l_rate*gra/ada
    print('iteration: %d | cost: %f  '%(i,cost_a))

# print(w)
# for i in range(9):
#     print(w[9*i+9])

# save model
np.save('model.npy',w)




