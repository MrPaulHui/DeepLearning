# softmax ce loss综合整理

## 公式

softmax公式：
$$
softmax(z_i)=\frac{e^{z_i}}{\sum_{j=1}^m e^{z_j}},\ i=1,2,...,m
$$
其中m为输出向量z的维度，即类别数

ce loss公式：
$$
ce\ loss = -\sum_i p(i)\log q(i)
$$
其中p为真实分布，q为预测分布，i为分布的维度，即类别数

## softmax计算过程中数值溢出问题

参考：https://blog.csdn.net/m0_37477175/article/details/79686164

softmax公式中，计算输出值可能会存在溢出，两种溢出：

1. 上溢出，$z_i$特别大的情况
2. 下溢出，$z_i$为负值，且$|z_i|$特别大，导致分母为极小的正数，可能会舍为0

解决办法：

令
$$
M=max(z_i),\ \ i=1,2,...,m
$$
计算出z的最大值，然后计算$softmax(z_i-M)$的值代替$softmax(z_i)$即可。

方法work的原因：

1. 针对上溢出，$z_i-M$的值小于等于0，$e^{z_i-M}$值小于等于1，不会出现$z_i$特别大的情况

2. 针对下溢出，至少有一个i，使$z_i-M$的值为0，$e^{z_i-M}$的值为1，就不会使分母过小

3. 相比于原式，现在式子为
   $$
   softmax(z_i-M)=\frac{e^{z_i-M}}{\sum_{j=1}^m e^{z_j-M}}\\
   =\frac{e^{z_i}/e^M}{(\sum_{j=1}^m e^{z_j})/e^M}\\
   =\frac{e^{z_i}}{\sum_{j=1}^m e^{z_j}}, \ i=1,2,...,m
   $$
   和原式结果没有区别

## KL散度与ce loss的关系

参考：https://zhuanlan.zhihu.com/p/74075915

KL散度：
$$
KL \ loss = \sum_i p(i)\log\frac{p(i)}{q(i)}
$$
其中p为真实分布，q为预测分布

KL散度衡量的是p和q两个分布的接近程度，二者越接近，KL loss越小

KL散度可以继续写成：
$$
KL \ loss = \sum_i p(i)\log\frac{p(i)}{q(i)}=\sum_i[p(i)\log p(i)-p(i)\log q(i)]
$$
可以发现第二项就是ce loss，第一项只有p分布，对于分类任务，真实分布p是确定的，所以第一项就是一个常数，无需保留在loss中。

而对于KD知识蒸馏，真实分布p是teacher model输出的分布，不是确定值，所以知识蒸馏训练中采用KL散度为损失函数，p为teacher model输出分布，q为student model输出分布，优化目标就是使二者接近。

## softmax反传

## 最优化角度看待softmax

参考：https://zhuanlan.zhihu.com/p/45014864