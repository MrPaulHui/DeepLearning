# 机器学习与深度学习常用的评价指标

## 分类

参考：

https://blog.csdn.net/zwqjoy/article/details/78793162

https://zhuanlan.zhihu.com/p/30153372

### 混淆矩阵

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy82NTQyNDc1LTM1NTdkM2Q0YmI5ODVhYWEucG5nP2ltYWdlTW9ncjIvYXV0by1vcmllbnQvc3RyaXAlN0NpbWFnZVZpZXcyLzIvdy80MTY)

TP：True Positive，实际的正类被预测为正类的，叫真正类

FN：False Negative，实际的正类被预测为负类的，叫假负类

FP：False Positive，实际的负类被预测为正类的，叫假正类

TN：True Negative，实际的负类被预测为负类的，叫真负类

可以发现，Positive和Negative都是针对预测的是正类还是负类所说的，True和False是指是否和真实的类别一致

以下所有指标的计算都是基于这个混淆矩阵

### 多分类情况

混淆矩阵针对的是二分类问题，包括下面的指标也是二分类情况下的，那么多分类情况呢？很简单，用1vsN就可以了，各自求每一类的指标，求这一类指标的时候，把这一类看做正类，其他所有类都看做负类。（acc不存在二分类还是多分类的问题）

### accuracy准确率

$$
acc=\frac{TP+TN}{TP+FN+FP+TN}
$$

最简单的，预测对的比上总数。缺点也很明显，对于类别不均衡的测试集，acc没有意义。比如100个样本，99个正类，1个负类，我的模型完全没有算法，单纯地预测所有的样本都为正类，对这个测试集可以达到99%的acc，这有意义吗？

### precision和recall

precision，查准率
$$
precision=\frac{TP}{TP+FP}
$$
即预测是正类的里面，实际是正类的所占的比重。表示预测的准不准。precison越大表示模型越好。

recall，召回率，查全率
$$
recall=\frac{TP}{TP+FN}
$$
即实际的正类中，预测出来多少。表示预测的全不全。recall越大表示模型越好。

### F1 Score

$$
F1=\frac{2}{\frac{1}{precision}+\frac{1}{recall}}
$$

同时考虑precision和recall，两个都要大，F1值才能大。

### ROC曲线和AUC面积

先引入两个新的指标，TPR，FPR

#### TPR

True Positive Rate，真正类率
$$
TPR=\frac{TP}{TP+FN}
$$
表示实际的正类中，预测对多少。越大越好。

#### FPR

False Positive Rate，假负类率
$$
FPR=\frac{FP}{FP+TN}
$$
表示实际的负类中，预测错多少。越小越好。

#### ROC曲线

以FPR为横轴，TPR为纵轴，绘制的曲线就是ROC曲线。

一个问题是FPR和TPR就一个坐标点，怎么画曲线？

这就涉及到分类阈值问题，分类问题输出的是属于每一类的概率（对于二分类，如果用sigmoid，那么输出的一个值就代表他属于正类的概率），对于这个概率，怎么判断他是正类还是负类呢？当然要设定阈值，大于这个阈值即判断预测为正类。那么设定不同的阈值，就可以得到不同的TPR和FPR二元组，也就是说每个阈值对应一个FPR和TPR坐标点，这样就可以有多个坐标点绘制成一条曲线。

![img](https://imgconvert.csdnimg.cn/aHR0cDovL3VwbG9hZC1pbWFnZXMuamlhbnNodS5pby91cGxvYWRfaW1hZ2VzLzUwODI4LWM0ZGZmZjYxZTNhZDU4ZmEucG5nP2ltYWdlTW9ncjIvYXV0by1vcmllbnQvc3RyaXAlN0NpbWFnZVZpZXcyLzIvdy8xMjQw)

曲线越往左上方凸，说明在相同的FPR下，TPR越大；相同的TPR下，FPR越小；即模型性能越好。

衡量往左上凸的程度最明显的就是用曲线下的面积，也就是AUC

#### AUC面积

即为ROC曲线下的面积，越大，说明模型性能越好。

#### 多分类的ROC曲线和AUC面积

设有C类，采用1vsN的思路，每一类可以单独计算出各自的TPR和FPR，各自的ROC，共有C个ROC曲线，取平均即可得到最终的ROC曲线，以及AUC。

需要注意，多分类采用softmax的输出，计算每一类的ROC时，该类的softmax输出值为样本属于这一类的概率，用这个值搭配不同的阈值，最终画出该类的ROC曲线。

### AP与mAP

#### PR曲线

以recall为横轴，precision为纵轴，通过不同阈值绘制曲线，称作PR曲线。两个坐标都是越大越好，所以曲线应该越往右上方凸越好，同样可以采用PR曲线下的面积来衡量，也就是AP，average precision。AP越大，模型性能越好。

#### mAP

mean average precision

AP是衡量二分类的，那么多分类呢？就是用mAP。

设有C类，采用1vsN的思路，每一类可以单独计算出各自的AP，共有C个AP，取平均即可得到mAP。

需要注意，多分类采用softmax的输出，计算每一类的AP时，该类的softmax输出值为样本属于这一类的概率，用这个值搭配不同的阈值，最终画出该类的PR曲线，进而得到AP值。

#### 目标检测里的mAP

见object detection具体

## 回归

mse，mae

## 聚类

见机器学习/ML/聚类.md

