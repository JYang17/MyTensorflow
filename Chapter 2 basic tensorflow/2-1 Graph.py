# https://www.bilibili.com/video/av20034351/?p=3

import tensorflow as tf
#创建一个1行2列常量
m1 = tf.constant([[3,3]])
#创建一个2行1列常量
m2 = tf.constant([[2],[3]])
#矩阵乘法，结果是向量（tensor）
product = tf.matmul(m1,m2)

#print(product)

#define a session to call the default graph
with tf.Session() as session:
    result = session.run(product)
    print(result)