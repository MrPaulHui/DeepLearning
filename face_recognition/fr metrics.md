# 人脸识别metrics

## 欧式距离，余弦距离，余弦相似度

设两个向量A，B，夹角为$\theta$

则余弦相似度为
$$
cos\_sim=\cos\theta
$$
余弦距离为
$$
D_{cos}=1-\cos\theta
$$
欧式距离为
$$
D_E=||A-B||_2=\sqrt{(A-B)^T(A-B)}=\sqrt{A^TA-A^TB-B^TA+B^TB}\\=\sqrt{||A||_2^2+||B||_2^2-2||A||_2||B||_2\cos\theta}
$$
若A，B经过了归一化，则$||A||=1,||B||=1$，所以归一化后的欧式距离为
$$
D_{E\_Norm}=\sqrt{2-2\cos\theta}
$$
可以发现，归一化后的欧式距离和余弦距离是等价的

**一般人脸识别中，都对embedding做normalize，所以用欧式距离和用余弦距离都是一样的。**

### 欧式距离和余弦距离的差异（高维上）

体现在高维上，无论维数多高，余弦距离的范围始终保持在$[-1,1]$，而欧式距离则不然，维度越高，数值越大，不具有数值稳定性，所以维度过高的时候，余弦距离（没有归一化的）可能会出现数值溢出。

## acc

$$
acc=\frac{识别对的对数}{总对数}
$$

## TAR FAR

参考：https://blog.csdn.net/liuweiyuxiang/article/details/81259492

设阈值为$T$

TAR，True Accept Rate，正确接受比例
$$
TAR=\frac{I(同人相似度>T)}{I(同人比较对数)}
$$
FAR，False Accept Rate，错误接受比例
$$
FAR=\frac{I(非同人相似度>T)}{I(非同人比较对数)}
$$

TAR越大，FAR越小，说明这个人脸识别模型越好。

但TAR和FAR的值都受阈值$T$的影响，极端情况下，若$T=0$，则$TAR=1$，$FAR=1$；若$T=1$，则$TAR=0$，$FAR=0$。所以一般测评的时候，**会设定一个固定的$FAR$值，看这个$FAR$值对应的$TAR$值，从而用这个值还衡量模型的好坏。**书写格式为：$TAR@FAR=1e-4$。一般$FAR$取1e-4，看具体要求。

**具体做法是根据设定好的$FAR$值，通过插值法找出对应的阈值，再通过这个阈值，计算出$TAR$值。**

## MegaFace

暂时不用

## attack test（公安部三所）

暂时不用