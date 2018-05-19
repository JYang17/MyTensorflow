# https://www.bilibili.com/video/av20034351/?p=4

import tensorflow as tf

x = tf.Variable([1,2])
a = tf.constant([3,3]) #注意是小写c，大小写敏感
#减法
sub = tf.subtract(x,a)
#加法
add = tf.add(sub,x)

#所以的变量都初始化，才能用
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    print(session.run(sub)) #注意是小写r，大小写敏感
    print(session.run(add))

#变量可以起名字
state = tf.Variable(0,name="counter")
new_state = tf.add(state,1)#注意调用add后state的值也加1了，而不是保持0
#tensorflow赋值不能直接用等号赋值
update = tf.assign(state, new_state)

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    print(session.run(state)) #注意是小写r，大小写敏感
    for _ in range(5):
        print(session.run(state))
        print(session.run(new_state))
        print(session.run(update))
        

#output
# 0
# 0
# 1
# 1
# 1
# 2
# 2
# 2
# 3
# 3
# 3
# 4
# 4
# 4
# 5
# 5