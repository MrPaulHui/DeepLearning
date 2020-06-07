# data parallel & model parallel

注：本文所讲的都是单机多卡版本

## 数据并行和模型并行概述

数据并行，将一个batch的数据平均划分到每一张卡上，而模型则是在每张卡上都复制一份，包括backbone和fc层（分类任务下）。

模型并行，backbone在每张卡上复制一份，fc层平均划分到每一张卡上，e.g. 一个80分类任务，输入到fc的feature map的size是512，那么需要一个80*512的fc层。如果是数据并行，这个80\*512的fc矩阵，在每张卡都要有一份复制；如果是模型并行，那么每张卡划分到的就是10\*512的矩阵（假设有8张卡），这样的好处就是大大节省了训练过程中的模型参数量和算力。

## 数据并行的实现

这个非常简单，都是深度学习库自带的，大概原理就是每部分数据经过各自卡上模型输出后，计算loss，然后将loss相加取平均后，广播到每张卡上，进行梯度回传。

## 模型并行的实现

设类别数为C，batch_size为B，embedding的维度为D，每张卡划分的类别数为EC

### 前传

backbone部分和数据并行是一样的，backbone在每张卡上都复制一份，做数据并行前传。将每张卡backbone输出的embedding做concat，在CPU上做concat，即得到一个batch所有的embedding。将concat后的embedding再加载每张卡上，现在每张卡上都有全部的batch数据，维度为B*D。

softmax前传的公式为
$$
softmax(z_i)=\frac{e^{z_i}}{\sum_{j=1}^m e^{z_j}},\ i=1,2,...,m
$$
现在每张卡上有一部分的fc层，维度为EC*D，batch数据通过该部分fc输出后，得到部分logit，维度为B\*EC，计算出$sum(e^{logit})$，所有卡的这个值加起来，就得到了softmax公式的分母项，同时实际编程中，需要logit值减去max(logit)，所以需要求出max(logit)，方法是先统计每张卡的max(logit)的，再得到全局的max。注意这里相加和统计全局max都是在CPU上进行。

在实际实现中，前传最重要的作用就是得到分母项，以及全局max。这里其实还可以算出loss，实际实现中是在反传中实现的。loss计算方法：找到每个样本实际对应的类别，计算$exp(logit_y)$，再除分母项，计算交叉熵损失。

### 反传

直接用推导出的softmax求梯度公式，

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

对于每张卡，先用第一个公式，计算出batch数据每个类别的梯度，也就是softmax输出，如果类别是某个样本的ground-truth class，那么根据第二个公式，做-1操作。然后把梯度回传到每个fc层，每个层的梯度在CPU上做concat之后，再回传给各个卡上的backbone，做反传。

