import csv
import numpy as np
import math

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