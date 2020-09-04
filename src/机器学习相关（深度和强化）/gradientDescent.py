import numpy    as np

def sigmoid(x):
    #sigmoid函数
    return 1/(1+np.exp(-x))
learnRate = 0.5
x = np.array([1,2])
y = np.array(0.5)

#起始权重
w = np.array([0.5,0.5])

#为每一个权重Wi 计算一个梯度下降
#计算神经网络的输出
nn_output = sigmoid(np.dot(x,w))

#计算神经网络的error
error = y - nn_output
#计算权重的改变
del_w = learnRate*error*nn_output*(1-nn_output)*x

print("神经网络输出")
print(nn_output)
print("error的数量：")
print(error)
print("change in weight:")
print(del_w)