# batchnorm系列

参考：

https://zhuanlan.zhihu.com/p/33173246

https://www.cnblogs.com/guoyaohua/p/8724433.html

## 基本运算

### 线性层后面的batchnorm

#### 前传

设输入进bn的数据为$X\in R^{N*C}$，其中N为batch size，C为数据维度

设X的行向量为$X_i,\ i=1,2,..,N$，$X_i\in R^{1*C}$，表示一个样本，

- 先求出样本各个维度的均值和方差

$$
\mu=\frac{1}{N}\sum_{i=1}^NX_i\\
\sigma^2=\frac{1}{N}\sum_{i=1}^N(X_i-\mu)^2
$$

$\mu\in R^{1*C}$，每个元素为样本对应的各个维度的均值，$\sigma^2\in R^{1*C}$，每个元素为样本对应的各个维度的方差

- 用均值和方差对样本进行标准化

$$
\hat X_i =\frac{X_i-\mu}{\sqrt{\sigma^2+\epsilon}},\ \ i=1,2,...,N
$$

其中$\epsilon$保证分母不为0

- 拉伸和平移

引入两个learnable的参数$\gamma,\beta$，其中$\gamma\in R^{1*C}$，$\beta\in R^{1*C}$，bn的输出为：
$$
Y_i=\gamma \hat X_i+\beta, \ \ i=1,2,...,N
$$
其中，$\gamma$和$X_i$的相乘为对应元素相乘

N个$Y_i$行向量组成矩阵$Y\in R^{N*C}$即为矩阵$X$通过bn输出的数据，二者维度是一致的

##### train和inference过程

bn运算过程中，需要用到均值和方差，在训练中，均值和方差是通过计算当前batch的数据得到的，那么测试中呢？测试中需要和训练中保持一致，就需要记录训练中的均值和方差数据，那那么多的batch，记录哪个呢？做法是采用**移动平均**相加，把每个batch的均值方差数据都加到两个变量moving_mean和moving_var，测试中数据通过bn层时，就用记录的这两个变量做运算。

移动平均的原理可参考SGD梯度下降法中的动量，这里具体的做法为
$$
moving\_mean=momentum*moving\_mean+(1-momentum)*now\_mean\\
moving\_var=momentum*moving\_var+(1-momentum)*now\_var
$$
now_mean和now_var为当前batch的均值和方差，原理就是把之前的moving值衰减一些，再加上当前的值，这也是指数平滑的原理。

#### 反传

设输出$Y$的梯度为$dY$，$dY\in R^{N*C}$，$dY_i$为$dY$的行向量

（下面的相乘都是逐元素相乘）

（参考op.md文章里链式求导）

中间变量的梯度：
$$
d\hat X_i=dY_i\frac{\partial Y_i}{\partial \hat X_i}=dY_i\ \gamma, \ \ d\hat X_i\in R^{1*C}
$$

$$
d\sigma^2=\sum_{i=1}^N d\hat X_i\frac{\partial \hat X_i}{\partial \sigma^2}=\sum_{i=1}^Nd\hat X_i(X_i-\mu)\frac{-1}{2}(\sigma^2+\epsilon)^{-\frac{3}{2}}
$$

$$
d\mu=\sum_{i=1}^N d\hat X_i\frac{\partial \hat X_i}{\partial \mu}+d\sigma^2\frac{\partial \sigma^2}{\partial \mu}=\sum_{i=1}^N d\hat X_i\frac{-1}{\sqrt{\sigma^2+\epsilon}}-d\sigma^2\frac{2}{N}\sum_{i=1}^N(X_i-\mu)
$$

注：这里$d\sigma^2$用所有$\hat X_i$的求和，原因是所有$\hat X_i$都有涉及到$\sigma^2$的运算，包括$\mu$也是

$X_i$的梯度：
$$
dX_i=d\hat X_i\frac{\partial \hat X_i}{\partial X_i}+d\mu\frac{\partial \mu}{\partial X_i}+d\sigma^2\frac{\partial \sigma^2}{\partial X_i}=d\hat X_i\frac{1}{\sqrt{\sigma^2+\epsilon}}+d\mu\frac{1}{N}+d\sigma^2\frac{2}{N}(X_i-\mu)
$$
拉伸参数$\gamma$的梯度：
$$
d\gamma=\sum_{i=1}^NdY_i\frac{\partial Y_i}{\partial \gamma}=\sum_{i=1}^NdY_i\hat X_i
$$
平移参数$\beta$的梯度：
$$
d\beta=\sum_{i=1}^NdY_i\frac{\partial Y_i}{\partial \beta}=\sum_{i=1}^NdY_i
$$

### 卷积层后面的batchnorm

#### 前传

设输入进bn的feature map为X，X的维度为$N*C*H*W$，其中，N为batch size，C为通道数，H和W为高和宽

相比于线性层，这里通道数C等价于线性层的维度C，设$X_i$为一个样本的feature map，$X_i\in R^{C*H*W},\ i=1,2,...,N$

- 求各个通道的均值和方差
  $$
  \mu=\frac{1}{N*H*W}\sum_{i=1}^N\sum_{h=1}^H\sum_{w=1}^W X_{ihw}\\
  \sigma^2=\frac{1}{N*H*W}\sum_{i=1}^N\sum_{h=1}^H\sum_{w=1}^W(X_{ihw}-\mu)^2
  $$
  其中，$X_{ihw}\in R^{C}$，为一个样本一个位置上所有通道元素组成的C维元素。

  $\mu\in R^{C}$，为各个通道的均值，是对$N*H*W$个元素取的均值，$\sigma^2\in R^{C}$，为各个通道的方差，也就是说，**在单个通道上，对$N*H*W$个元素同时做归一化**。

- 用均值和方差对样本进行标准化
  $$
  \hat X_{ihw} =\frac{X_{ihw}-\mu}{\sqrt{\sigma^2+\epsilon}}
  $$
  其中，$i=1,2,...,N,\ h=1,2,...,H,\ w=1,2,...,W$

- 拉伸和平移

  引入两个learnable的参数$\gamma,\beta$，其中$\gamma\in R^{C}$，$\beta\in R^{C}$，bn的输出为：
  $$
  Y_{ihw}=\gamma \hat X_{ihw}+\beta
  $$
  其中，$i=1,2,...,N,\ h=1,2,...,H,\ w=1,2,...,W$

##### 转换为线性层后面的bn

将feature map $X$ reshape为$(N*H*W)*C$，**将$(N*H*W)$看作batch size，就和线性层的bn一样了**，比较上面的前传过程，也可以发现是一样的。

#### 反传

先转换为线性层后面的bn，再按照线性层bn的反传运算即可。

## batchnorm的思想

参考：https://www.cnblogs.com/guoyaohua/p/8724433.html

### 针对的问题：Internal Covariate Shift

Internal Covariate Shift，ICS，内部分布偏移。

多层神经网络中，每一层的输出就是下一层的输入，而在训练过程中，每一层的参数在不停的变化，就导致这一层的输出分布也会不停变化，导致网络很难稳定的学习，因为网络还得去学习怎么适应分布的变化。

Covariate Shift的定义就是指ML系统实例集合<X,Y>中的输入值X的分布老是变，不符合IID假设，因为是分析网络内部，所以就是internal。

### BN基本出发点

针对ICS问题，BN的基本思想就是，**让每一层的输入也就是上一层的输出分布固定下来**。

## batchnorm的优点及work的原因

参考：https://www.cnblogs.com/guoyaohua/p/8724433.html

### 优点零 解决了ICS问题

### 优点一 防止梯度弥散，加速训练收敛

这是BN最主要的作用。

在训练过程中，如果不对每一层的输出加以规范约束，就容易导致**输出分布陷入激活函数的饱和区**，所谓激活函数饱和区，就是指在某段区域内，激活函数的值基本不发生变化，比如sigmoid激活函数定义域过大或过小的部分（参考op.md中的sigmoid），饱和区对应的激活函数梯度值十分接近于0（同样可参考op.md中的sigmoid），这就导致梯度回传过程中，到这一层梯度就特别小了，前面的层更新就特别缓慢，无法学习和收敛，产生梯度弥散现象。

加入BN之后，对每一层的输出就进行了规范，将输出值分布规范集中在0左右，是激活函数的非饱和区，反传过程中就会产生更大的梯度，从而加速收敛。所以搭建网络时，放的顺序就是线性层（卷积层）+BN+激活。

不过这里还有一个问题，如果每一层的输出经过BN后，全部落在激活函数的非饱和区，那么激活函数就退化成了线性函数，其非线性表征能力就没有了。针对这个问题的解决措施就是bn运算中的第三步，加入拉伸和平移参数，相当于把完全以0为均值1为方差的分布往旁边动了动，使得输出可以落入激活函数的饱和区，从而体现非线性。**这里追求的最佳状态是找到线性和非线性的一个平衡点，既能得到激活函数非线性表征能力，又避免分布太靠近饱和区导致梯度弥散**。

#### 关于加入拉伸和平移参数的另一种解释

上一层的线性层（卷积层）经过运算输出的结果，全部被粗暴的归一化到均值为0方差为1的分布，那么就会使线性层（卷积层）的运算失去意义，所以为了弥补这一点，就加入了这两个参数，**使得网络可以再次学习，以恢复上一层线性层（卷积层）的学习到的特征**。当然**更重要的意义还是恢复网络的非线性表征能力**。

#### 权重伸缩不变性角度

参考：https://zhuanlan.zhihu.com/p/33173246

这是从另一个角度来解释为什么BN能加速训练

现在已经基本不用sigmoid做激活函数了，取而代之的是relu，relu则没有sigmoid容易发生梯度弥散的特性。那么与relu结合的BN作用是什么呢？

还是从梯度值的大小为角度入手，线性层$Y=XW$，$dX=dY\ W^T$，X的梯度由上游梯度和W权重矩阵决定，X的梯度又影响其前面层的参数更新，如果权重W值过大就会产生梯度爆炸，过小就会产生梯度弥散。

输出Y经过BN后，
$$
BN(Y)=BN(XW)=\gamma\frac{XW-\mu}{\sigma}+\beta
$$
对权重矩阵乘上一个常数$\lambda$得到$W'=\lambda W$，有
$$
BN(Y')=BN(XW')=\gamma\frac{XW'-\mu'}{\sigma'}+\beta=\gamma\frac{\lambda XW-\lambda\mu'}{\lambda\sigma'}+\beta=\gamma\frac{XW-\mu}{\sigma}+\beta=BN(Y)
$$
可以发现，无论W变大或变小多少，经过BN后的输出都是相等的，就是**BN的权重伸缩不变性**
$$
\frac{\partial BN(XW')}{\partial X}=\frac{\partial BN(XW)}{\partial X}
$$
也就是说，**BN的输出关于上一层线性层的输入X的梯度，是对W权重矩阵的大小不敏感的**，W的大小不影响反向传播，从而就不会出现梯度弥散和梯度爆炸现象，从而加速了训练收敛。

### 优点二 有防止过拟合的正则化作用

参考：https://blog.csdn.net/CV_YOU/article/details/89416210

1. 从样本角度（数据角度）

   训练过程中，BN层会使一个batch内的所有样本关联在一起，即每个样本的输出都受到batch内所有样本的影响，相当于间接做了数据增强，达到减轻过拟合的作用。

2. 从特征角度（模型角度）

   BN层使输出的特征每个维度都服从类似的分布，使网络不会向一个特征过分偏移，也就减轻了过拟合。这点和dropout的防过拟合原理是类似的。（过拟合其实就是过分学习了某一些特征，过分学习了训练集在这一些特征上所体现出来的特性，而这些特性并不是网络表征的任务所需要的）

### 优点三 降低了对参数初始化的要求，可以使用大学习率

这个优点直接来自于BN的基本出发点，每一层都将输出固定到一定范围的分布，再作为下一层的输入，就对初始化的参数不过于敏感。

可以使用大学习率也是因为BN解决了ICS问题，ICS问题的存在使得只能谨慎设置小学习率，来保证网络的训练收敛。

## BN的缺点

参考：https://zhuanlan.zhihu.com/p/43200897

### 缺点一 batch size过小导致效果下降

如果batch size过小，那么得到的均值和方差就会有很大的随机性，即噪声太大，导致效果下降。一般batch size小于16，就不建议使用BN，小于8就会出现明显下降。

### 缺点二 对于像素级生成任务，BN效果不佳

如超分任务，风格迁移任务，使用BN反而导致效果下降。原因可能是在batch内，一个样本受到其他无关样本统计量的影响，弱化了这个样本本身特有的细节信息。（**注意看一下，超分比赛代码是没有用BN的**）

BN适用的是判别模型，比如图像分类，这是因为判别模型取决于数据的整体分布，BN做的就是这个。

### 缺点三 RNN等动态网络使用BN效果不好而且不方便

## 多卡训练的syncBN问题

参考：https://tramac.github.io/2019/04/08/SyncBN/

### 不做syncBN的问题

设batch size为B，GPU卡数为G，那么每张卡上的batch size为B/G，如果不做syncBN，在训练中每张卡上的数据通过BN时，用来计算均值和方差的batch size就是B/G，而不是我们想要的B，相当于batch size减小了，在小batch size下做BN则会导致效果下降。

### syncBN的做法——前传

**跨卡同步BN的核心就是求出所有卡上全部batch的均值和方差**，

记每张卡上输入BN的数据为$X_{gi},\ g=1,2,..,G,\ i=1,2,...,B/G$，先各自求出每张卡的样本和$S_g$以及样本平方和$S_g^s$，为
$$
S_g=\sum_{i=1}^{B/G}X_{gi}\\
S_g^s=\sum_{i=1}^{B/G}X_{gi}^2
$$
再进行同步，求出整个batch的样本和$S$以及样本平方和$S^s$，为
$$
S=\sum_{i=1}^GS_g\\
S^s=\sum_{i=1}^GS_g^s
$$
于是，整个样本的均值为
$$
\mu=\frac{S}{B}
$$
方差为
$$
\sigma^2=\frac{S^s}{B}-\mu^2
$$
最后将计算出的全局均值和方差分发到每张卡上，来进行BN运算。

BN运算中还有拉伸和偏移两个参数，syncBN后，每张卡应该共享这两个参数，即每张卡的拉伸参数都是一样的，偏移参数也都是一样的。这就意味着需要有相同的初始化，并且反传时候也要做同步，每张卡共享这两个参数的梯度值。

### syncBN的做法——反传

设每张卡上经过BN输出的数据为$Y_{gi},\ g=1,2,..,G,\ i=1,2,...,B/G$，通过上游梯度回传得到Y的梯度$dY_{gi}$，根据BN梯度反传的公式，计算$d\mu,d\sigma^2,d\gamma,d\beta$时，均需要batch内所有样本的dY信息（所有相加），而$dX_{gi}$又需要通过$d\mu,d\sigma^2$得到，所以$dX_{gi}$也需要全部样本的信息。

总之梯度反传时需要做一次同步，同步的是汇总每张卡上BN输出值的梯度，计算出$dX_{gi},d\gamma,d\beta$后，分发到每张卡上。

### mxnet与pytorch实现syncBN

留个坑

## 其他的norm方法

参考：https://blog.csdn.net/qq_41997920/article/details/89945972

### Layer Norm

对于维度为$N*C*H*W$的输入feature map $X$，在$C*H*W$维度上做归一化，也就是对一个单独的样本，计算这个样本全部维度的元素的均值和方差，和batch内其他样本没有联系。

如果是线性层，输入$X\in R^{N*C}$，就是对每个样本的C个维度做归一化。

类比BN的参数，$\mu,\sigma^2,\gamma,\beta$这四个参数都是标量，因为一个样本只求出来一个均值和一个方差，每个元素也是用相同的拉伸和偏移参数。

#### 优点

针对单个样本训练，不依赖其他数据，可以不受batch内其他数据的影响，适用于RNN等动态网络以及batch size过小的情况。

#### 缺点

对单个样本的每个维度（通道）做归一化，如果每个维度（通道）表征的是不同的含义，如一个维度表示大小，另一个表示颜色，那么将这两个维度做归一化，必然会影响性能。

### Instance Norm

对于维度为$N*C*H*W$的输入feature map $X$，在$H*W$维度上做归一化，对一个单独的样本，对其每个通道单独进行归一化，依然是只针对一个样本，和batch内其他样本没有联系。

线性层的Instance Norm是没有意义的（$H*W=1$）。

类比BN的参数，$\mu,\sigma^2,\gamma,\beta$都是$\in R^C$，每个通道专属一个参数，各个样本之间没有关系，有各自的参数。

#### 适用情况

适用于像素级的生成模型，解决了BN的第二个缺点

### Group Norm

介于Instance Norm和Layer Norm之间，也是针对单个样本的归一化，引入一个分组概念，将通道分组进行归一化。group=1，就是Instance Norm；group=C(通道数)，就是Layer Norm

