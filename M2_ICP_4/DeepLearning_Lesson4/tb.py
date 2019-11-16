import tensorflow as tf
a = tf.add(1,2)
b= tf.multiply(a,4)
c= tf.add(4,5)

d= tf.multiply(c,6)
e= tf.multiply(4,5)
f= tf.div(c,6)

g= tf.add(b,d)
h= tf.multiply(g,f)

with tf.Session() as sess:
 Writer = tf.summary.FileWriter("./logs", sess.graph)
 print(sess.run(h))