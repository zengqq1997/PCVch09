# PCVch09

python计算机视觉第九章实验

## 下载及安装

本章的目的是让你了解和运行 TensorFlow

> 安装环境
>
> 1. python3.6
> 2. win10

> 安装步骤
>
> 1. 在cmd中输入**pip3 install --upgrade tensorflow** ，直至安装完成。
> 2. 在python命令行中输入**import tensorflow，**如果不出现任何提示，则说明安装成功。

## MINIST

MNIST是深度学习的经典入门demo，他是由6万张训练图片和1万张测试图片构成的，每张图片都是28*28大小（如下图），而且都是黑白色构成（这里的黑色是一个0-1的浮点数，黑色越深表示数值越靠近1），这些图片是采集的不同的人手写从0到9的数字。

![img](https://images2017.cnblogs.com/blog/36692/201709/36692-20170902115518890-1918991710.png)

每一张图片包含28像素X28像素。我们可以用一个数字数组来表示这张图片：

![img](https://images2017.cnblogs.com/blog/36692/201709/36692-20170902115821374-1418821547.png)

可以看到，矩阵中有值的地方构成的图形，跟左边的图形很相似。之所以这样做，是为了让模型更简单清晰。特征更明显。

我们把这个数组展开成一个向量，长度是 28x28 = 784。如何展开这个数组（数字间的顺序）不重要，只要保持各个图片采用相同的方式展开。从这个角度来看，MNIST数据集的图片就是在784维向量空间里面的点, 并且拥有比较复杂的结构

### 下载

[Yann LeCun's MNIST page](http://yann.lecun.com/exdb/mnist/) 也提供了训练集与测试集数据的下载。

| 文件                                                         | 内容                                             |
| ------------------------------------------------------------ | ------------------------------------------------ |
| [`train-images-idx3-ubyte.gz`](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz) | 训练集图片 - 55000 张 训练图片, 5000 张 验证图片 |
| [`train-labels-idx1-ubyte.gz`](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz) | 训练集图片对应的数字标签                         |
| [`t10k-images-idx3-ubyte.gz`](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz) | 测试集图片 - 10000 张 图片                       |
| [`t10k-labels-idx1-ubyte.gz`](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz) | 测试集图片对应的数字标签                         |

### 部分实验代码

**构建卷积神经网络**

初始化神经网络的权重值，定义卷积层和池化层

```python
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)#正太分布的标准差设为0.1
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1],padding='SAME') # 保证输出和输入是同样大小
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
```

接下来我们可以开始实现第一层了。它由一个卷积接一个max pooling完成。卷积在每个5x5的patch中算出32个特征。卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。 而对于每一个输出通道都有一个对应的偏置量。

为了用这一层，我们把x_image变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)。我们把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling。

```python
w_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32]) 
x_image = tf.reshape(x,[-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1) 
h_pool1 = max_pool_2x2(h_conv1)                         
```

为了构建一个更深的网络，我们会把几个类似的层堆叠起来。第二层中，每个5x5的patch会得到64个特征。

```python
w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)# 输入的是第一层池化的结果
h_pool2 = max_pool_2x2(h_conv2)
```

现在，图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片。我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU。

```Python
w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1,7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
```

为了减少过拟合，我们在输出层之前加入dropout。我们用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout。 TensorFlow的tf.nn.dropout操作除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的scale。所以用dropout的时候可以不用考虑scale。

```python
keep_prob = tf.placeholder(tf.float32, name="keep_prob")# placeholder是占位符
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
```

最后，我们添加一个softmax层，就像前面的单层softmax regression一样。

```python
w_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2, name="y-pred")
```

模型建立好了，那么我们该如何去辨别这个模型的好坏呢？

主要通过这个模型对于手写字体的识别率

接下来的代码可以得出该模型的识别率

```python
ross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print "step %d, training accuracy %g"%(i, train_accuracy)
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print "test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
```

### 参数

**x（图片的特征值）**：这里使用了一个28*28=784列的数据来表示一个图片的构成，也就是说，每一个点都是这个图片的一个特征，这个其实比较好理解，因为每一个点都会对图片的样子和表达的含义有影响，只是影响的大小不同而已。

**y_（真实结果）：**来自MNIST的训练集，每一个图片所对应的真实值，如果是1，则表示为：[1 0 0 0 0 0 0 0 0]

**w_conv1（权重）**：这个值很重要，因为我们深度学习的过程，就是发现特征，经过一系列训练，从而得出每一个特征对结果影响的权重，我们训练，就是为了得到这个最佳权重值。

**b_conv1（偏置量）**：是为了去线性

**y_conv（预测的结果）**：单个样本被预测出来是哪个数字的概率，比如：有可能结果是[ 1.07476616 -4.54194021 2.98073649 -7.42985344 3.29253793 1.96750617 8.59438515 -6.65950203 1.68721473 -0.9658531 ]，则分别表示是0，1，2，3，4，5，6，7，8，9的概率，然后会取一个最大值来作为本次预测的结果，对于这个数组来说，结果是6（8.59438515）

### 完整实验代码

```python
import tensorflow as tf
import numpy as np # 习惯加上这句，但这边没有用到
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

sess = tf.InteractiveSession()

# 1、权重初始化,偏置初始化
# 为了创建这个模型，我们需要创建大量的权重和偏置项
# 为了不在建立模型的时候反复操作，定义两个函数用于初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)#正太分布的标准差设为0.1
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)


# 2、卷积层和池化层也是接下来要重复使用的，因此也为它们定义创建函数
# tf.nn.conv2d是Tensorflow中的二维卷积函数，参数x是输入，w是卷积的参数
# strides代表卷积模块移动的步长，都是1代表会不遗漏地划过图片的每一个点，padding代表边界的处理方式
# padding = 'SAME'，表示padding后卷积的图与原图尺寸一致，激活函数relu()
# tf.nn.max_pool是Tensorflow中的最大池化函数，这里使用2 * 2 的最大池化，即将2 * 2 的像素降为1 * 1的像素
# 最大池化会保留原像素块中灰度值最高的那一个像素，即保留最显著的特征，因为希望整体缩小图片尺寸
# ksize：池化窗口的大小，取一个四维向量，一般是[1,height,width,1]
# 因为我们不想再batch和channel上做池化，一般也是[1,stride,stride,1]
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1],padding='SAME') # 保证输出和输入是同样大小
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')


# 3、参数
# 这里的x,y_并不是特定的值，它们只是一个占位符，可以在TensorFlow运行某一计算时根据该占位符输入具体的值
# 输入图片x是一个2维的浮点数张量，这里分配给它的shape为[None, 784]，784是一张展平的MNIST图片的维度
# None 表示其值的大小不定，在这里作为第1个维度值，用以指代batch的大小，means x 的数量不定
# 输出类别y_也是一个2维张量，其中每一行为一个10维的one_hot向量，用于代表某一MNIST图片的类别
x = tf.placeholder(tf.float32, [None,784], name="x-input")
y_ = tf.placeholder(tf.float32,[None,10]) # 10列


# 4、第一层卷积，它由一个卷积接一个max pooling完成
# 张量形状[5,5,1,32]代表卷积核尺寸为5 * 5，1个颜色通道，32个通道数目
w_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32]) # 每个输出通道都有一个对应的偏置量
# 我们把x变成一个4d 向量其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数(灰度图的通道数为1，如果是RGB彩色图，则为3)
x_image = tf.reshape(x,[-1,28,28,1])
# 因为只有一个颜色通道，故最终尺寸为[-1，28，28，1]，前面的-1代表样本数量不固定，最后的1代表颜色通道数量
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1) # 使用conv2d函数进行卷积操作，非线性处理
h_pool1 = max_pool_2x2(h_conv1)                          # 对卷积的输出结果进行池化操作


# 5、第二个和第一个一样，是为了构建一个更深的网络，把几个类似的堆叠起来
# 第二层中，每个5 * 5 的卷积核会得到64个特征
w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)# 输入的是第一层池化的结果
h_pool2 = max_pool_2x2(h_conv2)

# 6、密集连接层
# 图片尺寸减小到7 * 7，加入一个有1024个神经元的全连接层，
# 把池化层输出的张量reshape(此函数可以重新调整矩阵的行、列、维数)成一些向量，加上偏置，然后对其使用Relu激活函数
w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1,7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# 7、使用dropout，防止过度拟合
# dropout是在神经网络里面使用的方法，以此来防止过拟合
# 用一个placeholder来代表一个神经元的输出
# tf.nn.dropout操作除了可以屏蔽神经元的输出外，
# 还会自动处理神经元输出值的scale，所以用dropout的时候可以不用考虑scale
keep_prob = tf.placeholder(tf.float32, name="keep_prob")# placeholder是占位符
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# 8、输出层，最后添加一个softmax层
w_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2, name="y-pred")


# 9、训练和评估模型
# 损失函数是目标类别和预测类别之间的交叉熵
# 参数keep_prob控制dropout比例，然后每100次迭代输出一次日志
cross_entropy = tf.reduce_sum(-tf.reduce_sum(y_ * tf.log(y_conv),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 预测结果与真实值的一致性，这里产生的是一个bool型的向量
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# 将bool型转换成float型，然后求平均值，即正确的比例
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 初始化所有变量，在2017年3月2号以后,用 tf.global_variables_initializer()替代tf.initialize_all_variables()
sess.run(tf.initialize_all_variables())

# 保存最后一个模型
saver = tf.train.Saver(max_to_keep=1)

for i in range(100):
    batch = mnist.train.next_batch(64)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1],keep_prob: 1.0})
        print("Step %d ,training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print("test accuracy %f " % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))



# 保存模型于文件夹
saver.save(sess,"save/model")

```

### 实验结果

当代码运行起来以后，我们发现，准确率大概在96%左右浮动。

![img](https://github.com/zengqq1997/PCVch09/blob/master/result.jpg)

接下来看看哪些容易出现错误的数字

![img](https://github.com/zengqq1997/PCVch09/blob/master/4.jpg)

![img](https://github.com/zengqq1997/PCVch09/blob/master/4error.jpg)

如图明明一眼能看出是4，结果根据预测值来看被识别成了6

![img](https://github.com/zengqq1997/PCVch09/blob/master/7.jpg)

![img](https://github.com/zengqq1997/PCVch09/blob/master/7error.jpg)

这个图人眼能看出是7，结果根据预测值被认为是4

![img](https://github.com/zengqq1997/PCVch09/blob/master/9.jpg)

![img](https://github.com/zengqq1997/PCVch09/blob/master/9error.jpg)

上图可能有点模糊的9，被识别成4

实验中还有很多类似的让人摸不着头脑的图，这里篇幅有限就不一一列举，有兴趣的同学可以去试试
