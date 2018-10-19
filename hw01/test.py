import csv
import numpy as np
from numpy.linalg import inv
import random
import sys
import math

# read model
w = np.load('model.npy')
print(w)

test_x = []
n_row = 0
text = open('data/test_X.csv','r')
row = csv.reader(text,delimiter=',')

# for r in row:
#      if n_row % 18 == 0:
#          test_x.append([])
#          for i in range(2,11):
#              test_x[n_row//18].append(float(r[i]))
#
#      else:
#          for i in range(2,11):
#              if r[i] != "NR":
#                  test_x[n_row//18].append(float(r[i]))
#              else:
#                  test_x[n_row // 18].append(float(0))
#      n_row = n_row + 1

# # 前9个小时作为feature
# for r in row:
#      if (n_row-9) % 18 == 0 and n_row!=0:
#          test_x.append([])
#          for i in range(2,11):
#              test_x[n_row//18].append(float(r[i]))
#      n_row = n_row + 1

# 前5个小时作为feature
for r in row:
     if (n_row-9) % 18 == 0 and n_row!=0:
         test_x.append([])
         for i in range(2,11):
             if i > 5:
                test_x[n_row//18].append(float(r[i]))
     n_row = n_row + 1


text.close()
# print(test_x)
test_x = np.array(test_x)

print(test_x.shape)
# 加偏置值b，W0*1就是b，所以在x0前面插入1
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x),axis=1)

ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append(a)

# print(ans)
filename = "result/predict.csv"
text = open(filename,"w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()


ans = []
n_row = 0
ans_text = open('data/ans.csv','r')
ans_row = csv.reader(ans_text,delimiter=',')

for r in ans_row:
    if n_row!=0:
        ans.append(r[1])
    n_row += 1

predict = []
ans_text = open('result/predict.csv','r')
predict_row = csv.reader(ans_text,delimiter=',')
for r in predict_row:
    predict.append(r[1])
text.close()


rmse = 0
ans = np.array(ans,dtype=int)
predict = np.array(predict,dtype=float)
# for i in range(len(predict)):
#     predict[i] = int(predict[i])
# print(predict[1])
# predict.astype(type(int))
print(ans.shape)
print(predict.shape)

err = ans-predict
square_of_err = np.dot(err.transpose(),err)
mse = square_of_err/len(ans)
rmse = math.sqrt(mse)
print("rmse = ",rmse)