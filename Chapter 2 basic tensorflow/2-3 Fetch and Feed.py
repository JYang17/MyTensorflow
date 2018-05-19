# https://www.bilibili.com/video/av20034351/?p=5
# Fetch 在一个会话中执行多个operation
import tensorflow as tf

#Fetch
input1 = tf.constant(3.0)
input2 = tf.constant(4.0)
input3 = tf.constant(5.0)

add = tf.add(input1, input2)
mul = tf.multiply(add,input3)

with tf.Session() as session:
    result = session.run([mul,add])
    print(result)

#Feed
#定义占位符，相当于指针
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2)

with tf.Session() as session:
    print(session.run(output,feed_dict={input1:[7.0],input2:[3.0]}))