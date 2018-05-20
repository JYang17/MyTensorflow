#https://www.bilibili.com/video/av20034351/?p=22
#用LSTM进行MNIST手写数字识别
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
#one-hot：将标签转化为的某一行，某一位是1，其他位都是0的形式
#"MNIST_data"创建一个当前程序路径下的文件夹，下载存放MNIST图片数据，也可以用绝对路径
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

#输入图片集28*28
n_inputs = 28 #输入一行，一行有28个数据
max_time = 28 #一共28行
lstm_size = 100 #一共100个隐藏单元
n_classes = 10 #10种分类
#每个批次大小
batch_size = 100
#计算一共有多少个批次,除完取整
n_batch = mnist.train.num_examples//batch_size

#placeholder,x对应的是0-9手写图像数据(images),y对应的是实际上的数字是多少(labels)
#对于[None,784]你可以想，每次输入1行，784列，会输入很多这样的1行
x = tf.placeholder(tf.float32,[None,784])#任意行，784列
y = tf.placeholder(tf.float32,[None,10])#任意行，10列（输出10个标签）

Weights = tf.Variable(tf.truncated_normal([lstm_size,n_classes],stddev=0.1))
#tf.zeros([1,10]和tf.zeros([10]是等效的，10列输出每列代表0-9中的一个识别选项
biases = tf.Variable(tf.constant(0.1,shape=[n_classes]))

#定义RNN网络
def RNN(X,weights,biases):
    inputs = tf.reshape(X,[-1,max_time,n_inputs])
    #定义LSTM基本cell
    #接口变了，https://stackoverflow.com/questions/45976234/module-tensorflow-contrib-rnn-has-no-attribute-basiclstmcell
    #lstm_cell = tf.contrib.rnn.core_rnn_cell.BasicLSMTCell(lstm_size)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    #final_state[0]是cell state
    #final_state[1]是hidden state
    #https://blog.csdn.net/mydear_11000/article/details/52414342
    outputs,final_state = tf.nn.dynamic_rnn(lstm_cell,inputs,dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1],weights)+biases)
    return results

#用softmax函数作为activation function，将计算出0-9每种选项的概率
prediction = RNN(x,Weights,biases)

#交叉熵代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=prediction))
#优化器
train = tf.train.AdamOptimizer(1e-4).minimize(loss)

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
