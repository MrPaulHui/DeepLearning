# 通用的损失函数

参考：

https://www.jiqizhixin.com/articles/2018-06-21-3

https://zhuanlan.zhihu.com/p/58883095

![img](https://image.jiqizhixin.com/uploads/editor/45bf00ba-ce0d-411e-a1f7-e685d68b98c8/1529558773847.png)

## 分类loss

### ce loss

#### softmax ce loss

多分类问题。

见op.md和softmax ce loss.md

#### sigmoid ce loss

二分类问题。对应二分类的交叉熵损失BCE loss

输出的logit值s，先经过sigmoid，转换为概率值y
$$
y=sigmoid(s)=\frac{1}{1+e^{-s}}
$$
设label为$\hat y$，$\hat y\in\{0,1\}$，损失函数为
$$
BCE\ \ loss=\frac{1}{N}\sum_{i=1}^N(\hat y_i\log y_i+(1-\hat y_i)\log(1-y_i))
$$

#### focal loss 难例挖掘版的ce loss

对于单个样本，其通过网络和softmax/sigmoid的输出向量为$p$，设$p_t$为属于ground-truth class的概率

原始的ce loss为
$$
ce\ \ loss=-\log (p_t)
$$
focal loss为
$$
focal \ \ loss = -(1-p_t)^\gamma\log(p_t)
$$
其中，$\gamma\ge 0$，为调制参数，是一个超参数。

focal loss的意义在于，给每个样本的loss加上一个对应的权重，这个权重由$p_t$决定。

如果一个样本的$p_t$很大，那么它的权重系数$ (1-p_t)^\gamma$会变得很小，即对于loss的贡献会很小，这样的意义在于，一个$p_t$很大的样本表明已经训练的很好了，属于easy sample了，无需再重点训练，理应对其施加更小的权重。

反之，如果一个样本的$p_t$很小，那么它的权重系数$ (1-p_t)^\gamma$会变得很大，对于loss的贡献会很大，因为$p_t$很小的样本属于难例，自然应该重点训练。

注意到，权重系数中还一个参数是调整系数$\gamma$，这个值同样决定了权重的大小。如果$\gamma=0$，那么就退化成原始的ce loss。$\gamma$越大，hard sample所占的权重就越大，easy sample所占的权重就越大，即$\gamma$越大，难例挖掘的程度就越大，就越注重对难例的训练。一般$\gamma$取2是最佳。

### KL散度loss

衡量两种分布之间的相似性，可以用来使一种分布靠近另一种分布。
$$
KL \ loss = \sum_i p(i)\log\frac{p(i)}{q(i)}=\sum_i[p(i)\log p(i)-p(i)\log q(i)]
$$
其中，p为真实分布即label（但可能是变化的，最典型的就是知识蒸馏），q为网络输出的分布。

ce loss是一种特殊KL散度loss，即真实分布p为固定值的时候，KL散度loss就是ce loss。

**KL散度损失最典型的应用就是知识蒸馏。**

### 指数损失

$$
exp\ \ loss = \exp[-yf(x)]
$$

其中，label $y\in\{-1,1\}$，为二分类问题所适用。

经典应用就是AdaBoost。可以参照模型集成.md。

缺点：对离群点、噪声敏感。

### hinge loss 折页损失

$$
hinge\ \ loss=max(0,1-yf(x))
$$

其中，label $y\in\{-1,1\}$，为二分类问题所适用。

折页损失的关键在于1这个margin，如果没有1，那么只要$f(x)$与$y$符号相同，就不会产生loss，但可能$f(x)$只比0大/小了一点点，没有足够的置信度，对于测试集而言可能会产生错误分类（具体可以参照SVM.md里的分析）。而加上1这个margin之后，假设$y=1$，那么对于$f(x)$的要求就不仅仅是大于0，而是要比0足够大，大得多，才可以不产生loss，这样就加上了足够大的置信度。

最经典的应用就是SVM。可以参照SVM.md。

hinge loss也有多分类版本，也参照SVM.md。

## 回归loss

### L1 loss、L2 loss、smooth L1 loss

注意一点，smooth L1 loss就是Huber loss。

L1 loss
$$
MAE=\frac{1}{N}\sum_{i=1}^N|y_i-f(x_i)|
$$
L2 loss
$$
MSE=\frac{1}{N}\sum_{i=1}^N(y_i-f(x_i))^2
$$
smooth L1 loss(Huber loss)
$$
smooth\ L1\ loss=\begin{cases}
\frac{1}{N}\sum_{i=1}^N(y_i-f(x_i))^2 & (\text{if}\quad |y_i-f(x_i)| \leq 1)\\
\frac{1}{N}\sum_{i=1}^N|y_i-f(x_i)| & (\text{otherwise})\\
\end{cases}
$$

#### L1和L2 loss在对于异常样本处理上的差异

对于一个异常样本$x$，其通过模型的输出$f(x)$肯定与label $y$有很大的差异，而L2 loss则会通过求平方进一步放大这个差异，所以如果用L2 loss来训练，会使模型偏向异常样本的方向梯度下降，从而影响整体模型的性能。**所以L2 loss受异常样本影响更大。**

#### L1、L2和Smooth L1在梯度上的问题

先注意一点，L1 loss对于输出$f(x)$的梯度始终为1。

在训练开始阶段，模型随机初始化，导致输出和label差距很大，这时候如果用L2 loss，则会产生过大的梯度，从而导致训练不稳定，而L1 loss则保持梯度为1，可以稳定的训练。所以**训练开始阶段，L1优于L2。**

在训练到后期时，输出和label已经十分接近，这时候如果用L1 loss，则梯度仍然为1，导致模型无法继续往精度更高的方向优化，而L2 loss则产生与差距值呈2倍关系的梯度，可以继续优化。所以**训练后期阶段，L2优于L1。**

smooth L1则综合了L1和L2的优点，进行分段，在输出和label差距值大的时候（对应训练开始阶段），使用L1 loss；差距值大的时候（对应训练后期），使用L2 loss。

### Log-Cosh Loss

$$
log-cosh\ loss=\frac{1}{N}\sum_{i=1}^N\log(\cosh(y_i-f(x_i)))
$$

其图象为

![img](https://image.jiqizhixin.com/uploads/editor/74081c19-c02f-420b-9655-54ea4bededc9/1529558774637.png)

**Log-Cosh Loss具有smooth L1 loss的所有优点**，即差距小时接近L2 loss，差距大时接近L1 loss。并且其二阶处处可微，可以用牛顿法。

## metric learning loss

### MarginRankingLoss

参考：https://juejin.im/post/6844903949124763656

输出两个值$x_1,x_2$，对这两个值需要做一个大小比较，对应的label为$y$，若$y=1$，则应有$x_1>x_2$，若$y=-1$，则反之。
$$
MarginRankingLoss=max(0,-y(x_1-x_2)+margin)
$$
这个损失通常用于：

1. GAN
2. 排名任务

### Triplet Loss

### Contrastive Loss



