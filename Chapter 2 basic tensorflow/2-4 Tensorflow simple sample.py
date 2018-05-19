# https://www.bilibili.com/video/av20034351/?p=6
import tensorflow as tf 
import numpy as np 

#制造实验数据
x_data = np.random.rand(100)
y_data = x_data * 0.1 + 0.2

#y是预测值，优化k和b使得其接近0.1,0.2
b = tf.Variable(0.)#大小写敏感，必须是大写V
k = tf.Variable(0.)
y = k * x_data + b

#二次代价函数,tf.square是取平方
loss = tf.reduce_mean(tf.square(y_data-y))
#梯度下降法优化器，learning rate 0.2
optimizer = tf.train.GradientDescentOptimizer(0.2)
#最小化误差
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()#别忘了()，否则有看不懂的error

with tf.Session() as sess:
    sess.run(init)
    for step in range(20):
        sess.run(train)
        if step%2 == 0:
            print(step,sess.run([k,b]))