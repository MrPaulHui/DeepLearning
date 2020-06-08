# op前传和反传

注：

1. 这里所说的梯度都是指损失函数相对于这个参数的偏导数，采用链式求导法则得到

2. 链式求导法则一个点，
   $$
   y=f(g(x),h(x))
   $$
   则y相对于x的梯度为
   $$
   dx=\frac{\partial y}{\partial g(x)}\frac{dg(x)}{dx}+\frac{\partial y}{\partial h(x)}\frac{dh(x)}{dx}
   $$

## softmax ce loss

这一层不涉及模型权重参数，位于反传最上游

### 前传

#### 标量形式

$$
softmax(z_i)=\frac{e^{z_i}}{\sum_{j=1}^m e^{z_j}},\ i=1,2,...,m
$$

$$
ce\ loss = -\sum_i p(i)\log q(i)
$$

即，
$$
softmax \ ce \ loss=-\log \frac{e^{z_y}}{\sum_{j=1}^m e^{z_j}}=-z_y+\log \sum_{j=1}^m e^{z_j}
$$
其中，$y$表示样本所属的类别（真实分布p只有1个为1，其余都为0，只能有一个类别）

#### 矩阵形式

### 反传

#### 标量形式

1. $i\neq y$（注意这里i表示的是类别维度）
   $$
   \frac{\partial \ loss}{\partial\ z_i}=\frac{e^{z_i}}{\sum_{j=1}^m e^{z_j}}\\
   =softmax(z_i)
   $$

2. $i=y$
   $$
   \frac{\partial \ loss}{\partial\ z_y}=-1+\frac{e^{z_y}}{\sum_{j=1}^m e^{z_j}}\\
   =-1+softmax(z_y)
   $$

#### 矩阵形式

## 线性层（全连接层）

### 前传

设输入向量为$X\in R^{N*D}$，N为batch size，D为向量维度；线性层权重为$W\in R^{D*C}$；则输出向量为$Y=XW$，$Y\in R^{N*C}$。

所以前传的矩阵形式即为
$$
Y=XW
$$
分开写每个矩阵的元素形式：
$$
X=\left[
\begin{matrix}
x_{11} & x_{12} & ... & x_{1D}\\
x_{21} & x_{22} & ... & x_{2D}\\
...\\
x_{N1} & x_{N2} & ... & x_{ND}
\end{matrix}
\right]
$$

$$
W=\left[
\begin{matrix}
w_{11} & w_{12} & ... & w_{1C}\\
w_{21} & w_{22} & ... & w_{2C}\\
...\\
w_{D1} & w_{D2} & ... & w_{DC}
\end{matrix}
\right]
$$

$$
Y=\left[
\begin{matrix}
y_{11} & y_{12} & ... & y_{1C}\\
y_{21} & y_{22} & ... & y_{2C}\\
...\\
y_{N1} & y_{N2} & ... & y_{NC}
\end{matrix}
\right]
$$

所以前传的标量形式为
$$
y_{11} = w_{11}x_{11}+w_{21}x_{12}+...+w_{D1}x_{1D}\\
...\\
y_{1C} = w_{1C}x_{11}+w_{2C}x_{12}+...+w_{DC}x_{1D}\\
...\\
y_{NC} = w_{1C}x_{N1}+w_{2C}x_{N2}+...+w_{DC}x_{ND}
$$

$$
y_{ij}=w_{1j}x_{i1}+w_{2j}x_{i2}+...+w_{Dj}x_{iD}\ \ i=1,2,...,N,\ \ j=1,2,...,C
$$

### 反传

设$Y$的梯度为$dY\in R^{N*C}$，（由上游的梯度反传得到的）
$$
dY=\left[
\begin{matrix}
dy_{11} & dy_{12} & ... & dy_{1C}\\
dy_{21} & dy_{22} & ... & dy_{2C}\\
...\\
dy_{N1} & dy_{N2} & ... & dy_{NC}
\end{matrix}
\right]
$$
求X的梯度：
$$
dx_{11}=dy_{11}\frac{\partial y_{11}}{\partial x_{11}}+dy_{12}\frac{\partial y_{12}}{\partial x_{11}}+...+dy_{1C}\frac{\partial y_{1C}}{\partial x_{11}}=dy_{11}w_{11}+dy_{12}w_{12}+...+dy_{1C}w_{1C}\\
dx_{12}=dy_{11}\frac{\partial y_{11}}{\partial x_{12}}+dy_{12}\frac{\partial y_{12}}{\partial x_{12}}+...+dy_{1C}\frac{\partial y_{1C}}{\partial x_{12}}=dy_{11}w_{21}+dy_{12}w_{22}+...+dy_{1C}w_{2C}\\
...\\
dx_{1D}=dy_{11}\frac{\partial y_{11}}{\partial x_{1D}}+dy_{12}\frac{\partial y_{12}}{\partial x_{1D}}+...+dy_{1C}\frac{\partial y_{1C}}{\partial x_{1D}}=dy_{11}w_{D1}+dy_{12}w_{D2}+...+dy_{1C}w_{DC}
$$

$$
dx_{1j}=dy_{11}\frac{\partial y_{11}}{\partial x_{1j}}+dy_{12}\frac{\partial y_{12}}{\partial x_{1j}}+...+dy_{1C}\frac{\partial y_{1C}}{\partial x_{1j}}=dy_{11}w_{j1}+dy_{12}w_{j2}+...+dy_{1C}w_{jC}
$$

其中，$j=1,2,...,D$
$$
dx_{ij}=dy_{i1}\frac{\partial y_{i1}}{\partial x_{ij}}+dy_{i2}\frac{\partial y_{i2}}{\partial x_{ij}}+...+dy_{1C}\frac{\partial y_{iC}}{\partial x_{ij}}=dy_{i1}w_{j1}+dy_{i2}w_{j2}+...+dy_{iC}w_{jC}
$$
其中，$i=1,2,...,N,\ \ j=1,2,...,D$

由此，可推出X梯度的矩阵形式为
$$
dX=dY\ W^T
$$
求W的梯度：
$$
dw_{ij}=dy_{1j}\frac{\partial y_{1j}}{\partial w_{ij}}+dy_{2j}\frac{\partial y_{2j}}{\partial w_{ij}}+...+dy_{Nj}\frac{\partial y_{Nj}}{\partial w_{ij}}=dy_{1j}x_{1i}+dy_{2j}x_{2i}+...+dy_{Nj}x_{Ni}
$$
其中，$i=1,2,...,D,\ \ j=1,2,...,C$

由此，可推出W梯度的矩阵形式为
$$
dW=X^T\ dY
$$

### 加入bias的情况

设偏置$b\in R^{N*1}$，前传为
$$
Y=XW+b
$$

$$
y_{ij}=w_{1j}x_{i1}+w_{2j}x_{i2}+...+w_{Dj}x_{iD}+b_i\ \ i=1,2,...,N,\ \ j=1,2,...,C
$$

反传中$dW$和$dX$和上面一样，求$db$如下
$$
db_i=dy_{i1}\frac{\partial y_{i1}}{\partial b_i}+dy_{i2}\frac{\partial y_{i2}}{\partial b_i}+...+dy_{iC}\frac{\partial y_{iC}}{\partial b_i}=dy_{i1}+dy_{i2}+...+dy_{iC}
$$

$$
db=dY\ I
$$

其中，$I\in R^{C*1}$，为全部是1的列向量，即
$$
I=\left[
\begin{matrix}
1\\
1\\
...\\
1
\end{matrix}
\right]
$$

## dropout

参考：https://zhuanlan.zhihu.com/p/38200980

实现细节：由于每个神经元在训练过程以概率p随机失活，而概率分布是参数为p的二项分布，期望为p，也就是说训练过程中每个神经元实际输出的值为pw（期望意义下），所以在inference的时候，输出也要乘上p，以保持和训练一致。更方便的做法是训练过程中直接除p，这样inference时候什么也不用做。

## 激活函数

参考：https://zhuanlan.zhihu.com/p/32610035

### 激活函数作用

在神经网络中引入非线性，如果没有激活函数的非线性表征能力，再多层的神经网络也只是相当于一个矩阵，即一个线性函数进行线性回归，不能表征任意的函数。

### sigmoid

#### 前传

设输入向量为$s$，输出向量为$y$，
$$
y=sigmoid(s)=\frac{1}{1+e^{-s}}
$$

#### 反传

设y的梯度为dy，（由上游的梯度反传得到）
$$
ds=dy\frac{e^{-s}}{(1+e^{-s})^2}
$$

#### 缺点

容易发生梯度弥散，

梯度弥散：梯度在反传过程中，幅度越来越小，导致反传回浅层的梯度特别小，导致浅层网络更新缓慢，学习不到有价值的特征。梯度弥散就是梯度消失

sigmoid容易梯度弥散的原因：

将sigmoid的导数继续推导为：
$$
\frac{e^{-s}}{(1+e^{-s})^2}=\frac{1}{e^s+2+e^{-s}}
$$
由于$e^{s}+e^{-s}>=2$，所以导数最大值为$\frac{1}{4}$，且当s趋近正无穷和负无穷时，导数都趋近为0。这就导致梯度反传至sigmoid这一层时，若s特别大，那么再往前面回传的梯度就很接近0，导致前面的层无法更新，即梯度弥撒了。