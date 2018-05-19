#https://www.bilibili.com/video/av20034351/?p=7
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

#生成实验数据
#从-0.5到0.5均匀分布的200个点（一维数据），np.linspace([-0.5,0.5,200])
#[:,np.newaxis]是为了给上面的一维数据加一个维度，也就是得到200行1列的数据
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]

noisy = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + noisy

#placeholder,相当于输入层是N行1列
x = tf.placeholder(tf.float32,[None,1])#[None,1]任意行，1列
y = tf.placeholder(tf.float32,[None,1])#[None,1]任意行，1列

#定义神经网络中间层，隐藏层就一层L1,1行10列
Weight_L1 = tf.Variable(tf.random_normal([1,10]))#注意是tf，而非np
biase_L1 = tf.Variable(tf.zeros([1,10]))#注意是tf，而非np
Wx_plus_b_L1 = tf.matmul(x,Weight_L1) + biase_L1#注意矩阵乘法有先后顺序
L1 = tf.nn.tanh(Wx_plus_b_L1)

#神经网络输出层，L2 10行1列
Weight_L2 = tf.Variable(tf.random_normal([10,1]))
biase_L2 = tf.Variable(tf.zeros([1,1]))#注意是[1,1]
Wx_plus_b_L2 = tf.matmul(L1,Weight_L2) + biase_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

#二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))#注意是y，而非y_data，要保证维度一致
#优化器
train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(2000):
        sess.run(train,feed_dict={x:x_data,y:y_data})

    #获取预测值
    prediction_value = sess.run(prediction,feed_dict={x:x_data})
    #画图
    plt.figure()
    plt.scatter(x_data,y_data)#,和y_data之间多一个空格都编译不过
    plt.plot(x_data,prediction_value,"r-",lw=5)
    plt.show()

