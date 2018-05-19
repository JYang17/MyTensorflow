#https://www.bilibili.com/video/av20034351/?p=9
#不使用卷积神经网络的MNIST
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
#import numpy as np 

#载入数据集
#one-hot：将标签转化为的某一行，某一位是1，其他位都是0的形式
#"MNIST_data"创建一个当前程序路径下的文件夹，下载存放MNIST图片数据，也可以用绝对路径
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

#每个批次大小
batch_size = 100
#计算一共有多少个批次,除完取整
n_batch = mnist.train.num_examples//batch_size

#placeholder,x对应的是0-9手写图像数据(images),y对应的是实际上的数字是多少(labels)
#对于[None,784]你可以想，每次输入1行，784列，会输入很多这样的1行
x = tf.placeholder(tf.float32,[None,784])#任意行，784列
y = tf.placeholder(tf.float32,[None,10])#任意行，10列（输出10个标签）

#创建一个没有隐藏层的神经网络
W = tf.Variable(tf.zeros([784,10]))
#[10,10]不行,tf.zeros([1,10]和tf.zeros([10]是等效的，10列输出每列代表0-9中的一个识别选项
b = tf.Variable(tf.zeros([1,10]))
#用softmax函数作为activation function，将计算出0-9每种选项的概率
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

#二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))
#梯度下降法优化器
train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

#tf.argmax(y,1)代表概率最大的对于那个位置，
#如果预测的和实际的概率最大位置一样，说明所识别的数字是一致的，就认为识别数字正确
correction_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))

#求准确率
#tf.cast(prediction,tf.float32)将bool转化为float就是准确率
#这个平均值应该是这波输入图片全集（不论训练集或测试集）的所以训练批次的平均值
accuracy = tf.reduce_mean(tf.cast(correction_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for step in range(21):#总共数据反复训练21次
        for batch in range(n_batch):#注意要加range，每批数据数据集分成多个批次
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batch_xs,y:batch_ys})
        
        #用测试集，进行测试
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Step:"+str(step)+" Accuracy:"+str(acc))
