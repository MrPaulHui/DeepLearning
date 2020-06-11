# 一些可能有点犯嘀咕的概念

1. 梯度反传是求网络所有参数关于loss的梯度的手段，本质就是链式求导法则。再说明白一点，设参数集合为$\{\theta_1,\theta_2,...,\theta_n \}$，共有n个参数，那么就要求n个梯度，分别为$\frac{\partial loss}{\part \theta_1},\frac{\partial loss}{\part \theta_2},...,\frac{\partial loss}{\part \theta_n}$，然后用最基本的SGD更新参数，就是$\theta_i=\theta_i-\frac{\part loss}{\part \theta_i},i=1,2,...,n$
2. 网络进行梯度反传完毕，**求出所有参数的梯度后**，优化器optimizer采用SGD或Adam等优化算法，利用各个参数的梯度信息更新所有的参数。
3. 深度学习三大要素：数据；网络结构（包括loss）；优化算法。
4. 训练过程就是优化loss，即求$argmin_\theta loss(\theta)$