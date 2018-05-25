# https://www.bilibili.com/video/av20034351/?p=19
#卷积神经网络手写数字识别
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

#安装tensorflow-gpu时遇见的坑：
# tensoflow-gpu version 1.5.0, pip install tensorflow-gpu==1.5.0
#tensorflow-gpu==1.7.0下，本代码也好用
#install CUDA 9.0 cuda_9.0.176_windows_network with patch 1 and patch 2
#安装CUDA 9.0后
#系统环境变量配置variable name: CUDA_PATH variable value: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0
#cmd下输入nvcc -V，来试环境变量配置正确否
#cudnn-9.0-windows7-x64-v7，注意CUDA和cudnn版本一定要完全匹配

#载入数据集
#one-hot：将标签转化为的某一行，某一位是1，其他位都是0的形式
#"MNIST_data"创建一个当前程序路径下的文件夹，下载存放MNIST图片数据，也可以用绝对路径
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

#每个批次大小
#ResourceExhaustedError (see above for traceback): OOM when allocating tensor with shape[10000,32,28,28] and type float on 
#batch_size = 100
batch_size = 10
#计算一共有多少个批次,除完取整
n_batch = mnist.train.num_examples//batch_size
#测试集每个batch大小
test_batch_size = 10
#测试集一共有多少个批次
n_test_batch = mnist.test.num_examples//test_batch_size

#初始化weight
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)#生成一个截断的正态分布
    return tf.Variable(initial)

#initialize biase
def biase_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#卷积层
def conv2d(x,W):
    #x:input：原矩阵? [batch,in_height,weight_height,in_channel]
    #W:filter：卷积核矩阵，与原矩阵进行矩阵点乘运算后，得到卷积后矩阵 
    # [filter_height,filter_width,in_channels,out_channels]
    #strides=[1,1,1,1],中间两个1分别是卷积核移动时横向和纵向的步长,strides[0]=strides[3]=1
    #padding="SAME"是代表计算卷积时边缘补0，计算卷积后新的矩阵维度与原矩阵一致
    #padding=“VALID”则会，丢失边缘处数据
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

#池化层pooling，这里用的是矩阵中元素的值最大者得以保留的策略，池化运算矩阵大小为2乘2
def max_pool_2x2(x):
    #x:input：原矩阵? [batch,in_height,weight_height,in_channel]
    #ksize=[1,2,2,1],第一个1估计是batch，中间2,2分别代表池化运算矩阵filter的长和宽，最后一个1估计是channel，黑白对应1，彩色RGB对应信道数3
    #strides中间两个2代表横向和纵向filter矩阵移动时步长,strides[0]=strides[3]=1
    #padding="SAME"是代表池化计算时，若不够一个filter矩阵大小，则边缘补0，计算卷积后新的矩阵维度与原矩阵一致
    #padding=“VALID”则会，若不够一个filter矩阵大小，则不补0，直接丢失边缘处数据，所以得到的池化后矩阵维度会小一圈
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

#placeholder,x对应的是0-9手写图像数据(images),y对应的是实际上的数字是多少(labels)
#对于[None,784]你可以想，每次输入1行，784列，会输入很多这样的1行
x = tf.placeholder(tf.float32,[None,784])#任意行，784列
y = tf.placeholder(tf.float32,[None,10])#任意行，10列（输出10个标签）

#先把每次输入1行，784列的输入数据，转化为一个矩形数组
#[-1,28,28,1]，[batch,in_height,in_width,in_channels],
# 值-1代表这个维度的值是需要动态计算的，这里对应于[None,784]中的None
x_image = tf.reshape(x,[-1,28,28,1])

#初始化第一个卷积层的权重和偏置值
#5乘5的卷积采样窗口，32个卷积核（卷积后新矩阵的厚度是32,32这个数好像是随意指定的吧？不是2的5次幂吧），
#从1个平面采样（指的是原矩阵x的厚度为1）
#这里多少个卷积核指的是原矩阵经过卷积运算后的“厚度”？卷积之前矩阵x_image的"厚度"是1,长和宽是28
W_conv1 = weight_variable([5,5,1,32])
#每个卷积核对应一个偏置
b_conv1 = biase_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第二个卷积层和池化层
#[5,5,32,64]经过第一次卷积后，矩阵厚度从1变成32，第二次卷积要使得厚度从32变成64
W_conv2 = weight_variable([5,5,32,64])
#每个卷积核对应一个偏置
b_conv2 = biase_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#x_image是1张（1张也就是厚度为1）28*28的图片，经过了第一次卷积和池化操作，变成了32张14*14的图片
#32张14*14的图片再经过第二次卷积和池化操作，变成了64张7*7的图片
#因为是padding="SAME"，因此卷积操作实际上不会改变图片的长和宽，改变长和宽的是每次卷积后的2*2池化的操作
#2*2步长为2的池化操作每次将图片长和宽除以2

#第一个全连接层
#7*7*64代表第二次池化后输出，1024代表指定有1024个全连接层的神经元
W_fcl1 = weight_variable([7*7*64,1024])
b_fcl1 = biase_variable([1024])
#将第二次池化层后的输出扁平化为1维
fcl_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fcl1 = tf.nn.relu(tf.matmul(fcl_flat,W_fcl1)+b_fcl1)

keep_prob = tf.placeholder(tf.float32)
h_fcl1_dropout = tf.nn.dropout(h_fcl1,keep_prob)

#第二个全连接层
#上个全连接层1024个神经元，第二次全连接层10个神经元，代表10种分类
W_fcl2 = weight_variable([1024,10])
b_fcl2 = biase_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fcl1_dropout,W_fcl2)+b_fcl2)

#交叉熵代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=prediction))
#adam优化
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
    for step in range(5):#总共数据反复训练21次
        for batch in range(n_batch):#注意要加range，每批数据数据集分成多个批次
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})
        
        #用测试集，进行测试
        #https://github.com/tensorflow/tensorflow/issues/136
        #可能GPU内存不够一下子加载所有测试集数据，所以也需要像训练集一样，分batch去feed
        # acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:0.7})
        # print("Step:"+str(step)+" Accuracy:"+str(acc))
        total_accuracy = 0.0
        for test_batch in range(n_test_batch):
            test_batch_xs,test_batch_ys = mnist.test.next_batch(test_batch_size)
            total_accuracy+=sess.run(accuracy,feed_dict={x:test_batch_xs,y:test_batch_ys,keep_prob:1.0})

        print("Step:"+str(step)+" Accuracy:"+str(total_accuracy/n_test_batch))
