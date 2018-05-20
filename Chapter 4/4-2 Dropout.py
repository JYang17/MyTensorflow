#https://www.bilibili.com/video/av20034351/?p=11
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
keep_prob = tf.placeholder(tf.float32)#dropout留存百分比

#用截断正态分布初始化比初始化为0,效果更好
# W = tf.Variable(tf.zeros([784,10]))
#定义2000维度是为了演示过拟合，一般过拟合就是因为网络太复杂，而数据量太小
W1 = tf.Variable(tf.truncated_normal([784,2000],stddev=0.1))
#[10,10]不行,tf.zeros([1,10]和tf.zeros([10]是等效的，10列输出每列代表0-9中的一个识别选项
b1 = tf.Variable(tf.zeros([1,2000])+0.1)
L1 = tf.tanh(tf.matmul(x,W1)+b1)
L1_dropout = tf.nn.dropout(L1,keep_prob)

W2 = tf.Variable(tf.truncated_normal([2000,1000],stddev=0.1))
b2 = tf.Variable(tf.zeros([1,1000])+0.1)
L2 = tf.tanh(tf.matmul(L1_dropout,W2)+b2)
L2_dropout = tf.nn.dropout(L2,keep_prob)

W3 = tf.Variable(tf.truncated_normal([1000,10],stddev=0.1))
b3 = tf.Variable(tf.zeros([1,10])+0.1)
L3 = tf.tanh(tf.matmul(L2_dropout,W3)+b3)
#用softmax函数作为activation function，将计算出0-9每种选项的概率
prediction = tf.nn.softmax(L3)

#对数似然代价函数/交叉熵，我对比了一下，比二次代价函数收敛的速度快
#因为对数似然代价函数后期的调整速度，与误差的大小成正比，所以收敛速度比二次代价函数快
# loss = tf.reduce_mean(tf.square(y-prediction))
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=prediction))

#优化器
#train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#1e-3代表10的-3次方，也可以用0.001
train = tf.train.AdamOptimizer(1e-3).minimize(loss)

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
    for step in range(5):#总共数据反复训练21次，训练5个周期示例一下，要么电脑太慢
        for batch in range(n_batch):#注意要加range，每批数据数据集分成多个批次
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.6})
        
        #用训练集，进行测试
        train_acc = sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels,keep_prob:0.6})
        #用测试集，进行测试
        test_acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:0.6})
        print("Step:"+str(step)+" Train Accuracy:"+str(train_acc)+" Test Accuracy:"+str(test_acc))
