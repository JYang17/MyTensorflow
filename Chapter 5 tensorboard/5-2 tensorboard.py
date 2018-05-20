#https://www.bilibili.com/video/av20034351/?p=15
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

#tensorboard中显示参数摘要
def VariableSummary(var):
    with tf.name_scope("variable_summary"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("variable_mean",mean)
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar("stddev",stddev)#标准差
        tf.summary.scalar("max",tf.reduce_max(var))#最大值
        tf.summary.scalar("min",tf.reduce_min(var))#最小值
        tf.summary.histogram("histogram",var)#直方图

#placeholder,x对应的是0-9手写图像数据(images),y对应的是实际上的数字是多少(labels)
#对于[None,784]你可以想，每次输入1行，784列，会输入很多这样的1行
with tf.name_scope("inputs"):#注意有冒号
    x = tf.placeholder(tf.float32,[None,784])#任意行，784列
    y = tf.placeholder(tf.float32,[None,10])#任意行，10列（输出10个标签）

#创建一个没有隐藏层的神经网络
#tip:1.ctrl+？可以注释或者取消注释 2.选中几行后按tab可缩进 3.ctrl+s保存 4.下一行要缩进的，上一行都是以冒号结尾
with tf.name_scope("layer"):#注意有冒号
    W = tf.Variable(tf.zeros([784,10]))
    VariableSummary(W)
    #[10,10]不行,tf.zeros([1,10]和tf.zeros([10]是等效的，10列输出每列代表0-9中的一个识别选项
    b = tf.Variable(tf.zeros([1,10]))
    #用softmax函数作为activation function，将计算出0-9每种选项的概率
    prediction = tf.nn.softmax(tf.matmul(x,W)+b)

#对数似然代价函数/交叉熵，我对比了一下，比二次代价函数收敛的速度快
#因为对数似然代价函数后期的调整速度，与误差的大小成正比，所以收敛速度比二次代价函数快
# loss = tf.reduce_mean(tf.square(y-prediction))
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=prediction))
    tf.summary.scalar("loss",loss)

#梯度下降法优化器
with tf.name_scope("train"):
    train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

#tf.argmax(y,1)代表概率最大的对于那个位置，
#如果预测的和实际的概率最大位置一样，说明所识别的数字是一致的，就认为识别数字正确
with tf.name_scope("correct_prediction"):
    correction_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
    # tf.summary.scalar("correct_rate",correction_prediction)

#求准确率
#tf.cast(prediction,tf.float32)将bool转化为float就是准确率
#这个平均值应该是这波输入图片全集（不论训练集或测试集）的所以训练批次的平均值
with tf.name_scope("accuracy"):
    accuracy = tf.reduce_mean(tf.cast(correction_prediction,tf.float32))
    tf.summary.scalar("accuracy",accuracy)

#合并所有的跟踪的summary
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    #在当前程序路径下的Logs文件夹，保存graph信息
    #本程序在Python下运行后，会生成Logs文件夹和下面的文件
    #在D盘或者Logs文件夹下或vscode中在Logs文件夹右键“在命令提示符中打开”，
    # 运行cmd，输入tensorboard --logdir="d:\tensorflow sample\Logs"
    # d:\tensorflow sample\Logs>tensorboard --logdir="d:\tensorflow sample\Logs"
    # TensorBoard 1.5.1 at http://FQKGGDNHWPCERHA:6006 (Press CTRL+C to quit)
    #在Chrome浏览器中打开得到的url http://FQKGGDNHWPCERHA:6006 即可
    #要是有问题，结束访问Logs的cmd，删除Logs文件夹再试试
    writer = tf.summary.FileWriter("Logs",sess.graph)
    for step in range(11):#总共数据反复训练11次
        for batch in range(n_batch):#注意要加range，每批数据数据集分成多个批次
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            #运行时会同时运行merged和train，把merged结果存到summary
            summary,_=sess.run([merged,train],feed_dict={x:batch_xs,y:batch_ys})
        
        #输出summary到文件
        writer.add_summary(summary,step)

        #用测试集，进行测试
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Step:"+str(step)+" Accuracy:"+str(acc))
