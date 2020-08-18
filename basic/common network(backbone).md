# 常用的网络结构

## inception

参考：

https://blog.csdn.net/yuanchheneducn/article/details/53045551

[http://chenzhen.online/2019/04/11/%E6%B7%B1%E5%BA%A6%E5%8D%B7%E7%A7%AF%E7%BD%91%E7%BB%9C-Inception%E7%B3%BB%E5%88%97/](http://chenzhen.online/2019/04/11/深度卷积网络-Inception系列/)

### v1

![这里写图片描述](https://img-blog.csdn.net/20161108152543838)

1\*1卷积核的作用是改变channel数

inception的优点在于多尺度的表征能力

### v2

对v1的改进：

1. 加入BN（应该是每个输出的feature map之后）
2. 用两个3\*3的卷积核取代5\*5的卷积核

![img](https://res.cloudinary.com/chenzhen/image/upload/v1554957713/github_image/2019-04-11/Inception_V2.png)

### v3

改进：将n\*n的卷积核分解为1\*n和n\*1的卷积核

这样改进的好处：

1. 加速计算
2. 增加网络深度和非线性

### v4

和resnet相结合

## mobilenet

### 深度可分离卷积

![img](https://pic3.zhimg.com/80/v2-3060b36fe063bbfe99622be4a4b23a02_720w.jpg)

设输入的feature map size为$D_F*D_F*M$，期望输出的feature map size为$D_F*D_F*N$

- 普通卷积

  卷积核size为$D_k*D_k*M$，个数为$N$，参数量为$D_k*D_k*M*N$

- 深度可分离卷积

  1. depthwise conv

     每个channel对应一个size为$D_k*D_k*1$卷积核，M个channel对应M个这样size的卷积核

     所以depthwise这一步的参数量为$D_k *D_k *M$

  2. pointwise conv

     depthwise conv得到了一个$D_F*D_F*M$的feature map，但期望的feature map是N个channel的，所以用1\*1卷积来改变channel数，即用N个size为$1*1*M$的卷积核来得到size为$D_F*D_F*N$的期望feature map

     所以pointwise这一步的参数量为$N*M$

  深度可分离卷积的总参数量为$D_k *D_k *M+N*M$

对普通卷积核深度可分离卷积的参数量进行比较
$$
\frac{D_k *D_k *M+N*M}{D_k*D_k*M*N}=\frac{1}{N}+\frac{1}{D_k^2}
$$
一般卷积核大小$D_ k$为3，且channel N都很大，N越大，比例值越小，深度可分离卷积的参数量相比普通的就越小，当N远大于$D_k$时，比例值可近似为$\frac{1}{9}$，即最小可达到普通的的$\frac{1}{9}$。

## resnet

## densenet

## senet

