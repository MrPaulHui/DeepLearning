# optimizer

参考：动手学深度学习——优化算法

深度学习中优化的本质就是最小化损失函数，采用的方法就是梯度下降法，求梯度的方法就是反向传播。

## SGD

一阶梯度下降。

### BGD

batch gradient descent，批量梯度下降，最基础的梯度下降。

设训练集为$\{X_i\},\ i=1,2,...,N$，模型参数为$\theta$（为向量，包括所有的参数），损失函数为$loss(\theta;X_i)$

BGD的做法是每次迭代，遍历所有样本，求出所有样本的loss，求其对$\theta$的梯度
$$
\nabla \theta=\frac{\partial \frac{1}{N}\sum_{i=1}^Nloss(\theta;X_i)}{\partial \theta}
$$
更新参数$\theta$：
$$
\theta=\theta-\alpha \nabla\theta
$$
其中，$\alpha$为学习率，控制一次更新的步幅

#### 优缺点

- 优点

  得到的是最准确的梯度，最准确的下降方向

- 缺点

  1. 如果下降中陷入了鞍点（loss关于变量$\theta$的鞍点），就无法逃离鞍点，优化将停止，无法收敛到极小值点。

     原因：BGD每次迭代，都是使用的全部训练样本，这就导致

  2. 

### SGD

stochastic gradient descent，随机梯度下降。

设训练集为$\{X_i\},\ i=1,2,...,N$，模型参数为$\theta$（为向量，包括所有的参数），损失函数为$loss(\theta;X_i)$

SGD的做法是每次迭代，只取训练集中一个样本，求出这个样本的loss，求其对$\theta$的梯度
$$
\nabla\theta=\frac{\partial loss(\theta;X_i)}{\partial \theta}
$$
更新参数$\theta$：
$$
\theta=\theta-\alpha \nabla\theta
$$

#### 优缺点

### mini-batch SGD

小批量随机梯度下降。

设训练集为$\{X_i\},\ i=1,2,...,N$，模型参数为$\theta$（为向量，包括所有的参数），损失函数为$loss(\theta;X_i)$

mini-batch SGD的做法是，每次迭代，选取训练集中一个小批量的样本，设batch size为B，求出这B个样本的loss，求其对$\theta$的梯度
$$
\nabla\theta=\frac{\partial \frac{1}{B}\sum_{i=1}^Bloss(\theta;X_i)}{\partial \theta}
$$
更新参数$\theta$：
$$
\theta=\theta-\alpha \nabla\theta
$$

#### 优缺点

### 总结

以上三种梯度下降法，区别是每次迭代用多少样本来求loss对参数的梯度，更新参数的方法都是一样的，即最基本的梯度下降。

一般说的SGD都是指mini-batch SGD。

### 带动量的SGD

对于求出的参数梯度$\nabla\theta_t$，不直接用其来更新参数，而是先做一步
$$
v_t=\gamma v_{t-1}+\alpha\nabla \theta_t
$$
其中，$\alpha$为学习率，$\gamma$为动量

再更新参数$\theta$：
$$
\theta_t=\theta_{t-1}-v_t
$$

#### 指数加权移动平均

基本思想：当前时间步的变量，由上一个时间步的变量以及这个时间步另一个变量共同决定，组合方式为
$$
y_t=\gamma y_{t-1}+(1-\gamma)x_t
$$
其中，$0=<\gamma<1$

将上式进行展开，有
$$
\begin{equation}
\begin{aligned}
y_t & =(1-\gamma)x_t+\gamma y_{t-1}\\
& = (1-\gamma)x_t+\gamma ((1-\gamma)x_{t-1}+\gamma y_{t-2})\\
& = (1-\gamma)x_t+(1-\gamma)\gamma x_{t-1}+\gamma^2y_{t-2}\\
& = (1-\gamma)x_t+(1-\gamma)\gamma x_{t-1}+(1-\gamma)\gamma^2x_{t-3}+\gamma^3y_{t-3}\\
& = \ ...\\
& = (1-\gamma)\sum_{i=0}^N\gamma^ix_{t-i}+\gamma^{N+1}y_{t-(N+1)}
\end{aligned}
\end{equation}
$$
令
$$
n=\frac{1}{1-\gamma}
$$
则，
$$
(1-\frac{1}{n})^n=\gamma^{\frac{1}{1-\gamma}}
$$

$$
\lim_{\gamma\to1}\gamma^{\frac{1}{1-\gamma}}=\lim_{n\to\infty}(1-\frac{1}{n})^n=e^{-1}
$$

$e^{-1}$可以近似看做无穷小，所以，$\gamma\to1$时，可以将$\gamma^{\frac{1}{1-\gamma}}$看做无穷小，所以就可以忽略$\gamma^{\frac{1}{1-\gamma}}$以及比$\gamma^{\frac{1}{1-\gamma}}$更高阶的无穷小。

如取$\gamma=0.95$，$\frac{1}{1-\gamma}=20$，那么在$y_t$的展开式中，就可以忽略$0.95^{20}$及更高阶的项，也就是取N=19，可以把$y_t$写为
$$
y_t=0.05\sum_{i=0}^{19}0.95^ix_{t-i}
$$
所以，当$\gamma\to1$时，$y_t$为
$$
y_t=(1-\gamma)\sum_{i=0}^{\frac{1}{1-\gamma}-1}\gamma^ix_{t-i}
$$
也就是说，$y_t$的值是过去$\frac{1}{1-\gamma}$个时间步$x_t$值的加权和，且越靠近当前时间步，所占权重越大。

#### 指数加权移动平均用于动量SGD

动量SGD先对梯度做一步操作：
$$
v_t=\gamma v_{t-1}+\alpha\nabla \theta_t
$$
可以写成
$$
v_t=\gamma v_{t-1}+(1-\gamma)(\frac{\alpha}{1-\gamma}\nabla\theta_t)
$$
根据指数加权移动平均，当$\gamma\to1$时，$v_t$可写成
$$
v_t=(1-\gamma)\sum_{i=0}^{\frac{1}{1-\gamma}-1}\gamma^i\frac{\alpha}{1-\gamma}\nabla\theta_{t-i}=\alpha\sum_{i=0}^{\frac{1}{1-\gamma}-1}\gamma^i\nabla\theta_{t-i}
$$
可以发现，$v_t$为过去$\frac{1}{1-\gamma}$个时间步$\nabla \theta_t$的的加权和，再乘上学习率，这就意味着**用来更新当前参数所用的梯度，不仅仅是当前时间步的梯度，还考虑了过去多个时间步的梯度**。

#### 动量SGD的优点

传统的SGD每次迭代，都是只用当前batch计算得到的梯度，来进行梯度下降，这样存在的问题是如果某一个参数的梯度特别大，那么更新后，就会向这个参数方向大幅度的下降，这个大幅度有可能就会导致越过了最优解的位置，从而无法收敛。针对这个问题，可以采用设置小学习率来解决，但是小学习率又会导致收敛速度变慢。

动量SGD就是针对这个问题，动量SGD每次更新，不仅用当前batch的梯度，同时还考虑过去多个时间步的梯度，**这样在遇到上面所说的某一个参数的梯度特别大的情况时，由于前面多个时间步梯度的累积的影响，当前梯度和前面累积的梯度做加权和后，就会减小这个参数方向的下降幅度，从而不会向这一个参数方向大幅度下降**。

**前面累积的梯度代表了训练中一个比较稳定的下降趋势**，而当前batch计算出的梯度是有一定的随机性，用累积的梯度和当前梯度做加权和，就会**减轻随机性带来的影响**，如向某一个方向大幅倾斜，loss震荡等，从而**使整个训练过程保持稳定性**。

### weight deacy

就是L2Norm，在loss中加入参数权重的二范数，目的是防止过拟合。

## AdaGrad

Ada就是自适应的意思，指的是**各个维度的参数都有自己自适应的学习率**。

SGD中，所有维度的参数都使用同样的学习率来更新，这样就会有在动量SGD里所说的问题，即可能某些维度的参数梯度过大，另一些过小。SGD采用加入动量即利用前面累积梯度的方法来解决，AdaGrad则是对每个维度的参数自适应的设置学习率。

对于求出的当前时间步的参数梯度$\nabla\theta_t$，做如下运算：
$$
s_t=s_{t-1}+(\nabla\theta_t)^2
$$
其中，平方运算是逐元素运算，$s_t$的维度和$\nabla\theta_t$的维度是一致的，

再更新参数：
$$
\theta_t=\theta_{t-1}-\frac{\alpha}{\sqrt{s_t+\epsilon}}\nabla\theta_t
$$
其中，$\epsilon$为数值稳定作用。$\alpha$为学习率。

所有运算都是逐元素的运算，这就说明每个维度的参数都有各自不同的学习率。

$s_t$是各个维度参数梯度的累积，$s_t$越大，说明这个维度的参数的梯度一直都比较大，体现在学习率上，这个维度参数的学习率就下降的快（因为更新公式中，学习率与$s_t$成反比）；$s_t$越小，这个维度参数的学习率就下降的慢。这样的机制就会使更新方向不会向梯度大的参数过分倾斜，保持下降趋势的稳定性。

### 缺点

随着$s_t$的一直累积，意味着每一维度参数的学习率都是在一直减小（或不变）的，训练前期梯度都较大，所以学习率会下降得很快，如果这个阶段不能收敛到一个比较好的解，那么在后期，学习率过小就会导致训练不动。

## RMSProp

RMSProp算法是在AdaGrad算法基础上的改进，改进的点就是AdaGrad的缺点，即前期学习率下降太快导致后期训练不动。

**改进的方法是对$s_t$的累积方式进行修改，加入动量**：
$$
s_t=\gamma s_{t-1}+(1-\gamma)(\nabla\theta_t)^2
$$
加入动量之后，$s_t$就不再是全部时间步的梯度平方和的累积，而是最近$\frac{1}{1-\gamma}$个时间步的梯度平方和加权累积，这样$s_t$就不会一直增大，学习率也不会一直减小。

参数更新的公式是和AdaGrad一样的。

## Adam

Adam在RMSProp基础上继续改进，改进的点是**对梯度本身也加入了动量**
$$
v_t=\gamma_1 v_{t-1}+(1-\gamma_1)\nabla\theta_t
$$
和RMSProp一样，$s_t$也加入动量
$$
s_t=\gamma_2 s_{t-1}+(1-\gamma_2)(\nabla\theta_t)^2
$$
第二个改进点是**对$v_t$和$s_t$进行偏差修正**

根据指数加权移动平均的公式，将$v_t$展开，并且有初始化条件为$v_0=0$
$$
v_t=(1-\gamma_1)\sum_{i=0}^{t-1}\gamma_1^i\nabla\theta_{t-i}+\gamma_1^tv_0=(1-\gamma_1)\sum_{i=0}^{t-1}\gamma_1^i\nabla\theta_{t-i}
$$
$v_t$的值由前面t个时间步的梯度加权得到，考虑这t个梯度的权重之和为
$$
(1-\gamma_1)\sum_{i=0}^{t-1}\gamma_1^i=1-\gamma_1^t
$$
同理，$s_t$的值由前面t个时间步的梯度平方和加权得到，这t个梯度平方和的权重之和为$1-\gamma_2^t$

t越小，这两个权重之和就越小，这就会导致训练前期和后期出现偏差，为了修正这个偏差，做法就是归一化，即对每个时间步的$v_t$和$s_t$都除各自的权重之和
$$
\hat v_t=\frac{v_t}{1-\gamma_1^t}\\
\hat s_t=\frac{s_t}{1-\gamma_2^t}
$$
参数更新公式和RMSProp是一样的，只不过用$\hat s_t$代替$s_t$，$\hat v_t$代替$\nabla\theta_t$，
$$
\theta_t=\theta_{t-1}-\frac{\alpha}{\sqrt{\hat s_t+\epsilon}}\hat v_t
$$
其中，$\alpha$为学习率

可以说Adam是RMSProp和动量法的结合。



## 牛顿法

留个坑。

## 鞍点问题

留个坑。

