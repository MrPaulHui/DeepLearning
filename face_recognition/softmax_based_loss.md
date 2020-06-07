# softmax based loss

参考：

https://louishsu.xyz/2019/07/13/SphereFace-CosFace-ArcFace/

https://zhuanlan.zhihu.com/p/60747096

[微信公众号的一篇](https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247493329&idx=1&sn=81e23cd23de94802a2ebdcf3ff3ca77b&chksm=ec1c0b28db6b823e505778eee6afc6639f660dd6ad1bf1da529bee4ac6cee4631b3b90bf32bc&mpshare=1&scene=24&srcid=&sharer_sharetime=1580827750166&sharer_shareid=062bb582160e3d3f23fa22b2ef2d352f&key=18ee9261cab3fde088668b99b574bfc9ca1fb44780c37716d4b68ea6f7e3d525ccdcd1ffba1dd97da96e85a67383d4f1dfca7fc6017adf70a623f83de86766601a40935fbec4122d63c12f5ddb25dba0&ascene=14&uin=MjExNDk1MTAwOA%3D%3D&devicetype=Windows+10+x64&version=62090070&lang=zh_CN&exportkey=A0qXQvDCey8oPySgh1huQkg%3D&pass_ticket=XBsSJKZJiU1w4uw95vv7RiNw%2BSth%2FaeylW02SqigRE4toIGI4qPi40tcqavcuE6L)

## norm归一化

人脸识别中backbone（或者说backbone和head，GluonFace的说法）输出的是embedding feature，后面做identification和verification就是用这个embedding。但是在训练过程中，如果用softmax based loss做监督信号，那么这个embedding还要再接一个fc层，从而输出类别分数，用softmax ce loss监督训练。

问题就出在这个fc层上，输出的类别分数logit，计算公式为：
$$
logit = embedding.dot(W)
$$
其中，设embedding的维度为$N*D$（D一般取256,512），N为样本数，fc层的权重W维度为$D*C$，C为训练集类别数，logit的维度为$N*C$。

计算公式还可以写成：
$$
logit = embedding.dot(W)=||embedding||\ ||W||\ cos(\theta)
$$
其中，$\theta$为embedding和W类中心（对应类别的向量）的夹角。

这个式子中，我们的目标是只优化余弦相似度$\cos \theta$，因为我们是用余弦相似度来表征一个样本与其类中心的靠近程度，余弦相似度越大，说明越靠近，说明这个类的类内距离越小。（类间距离则体现在fc层权重W每个向量之间的离散程度，这也是训练中的一个优化方向）。logit值通过softmax后，计算ce loss。

所以说，最终loss中，不仅有余弦相似度，还有两个范数——$||embedding||$和$||W||$，这两个对模型是没有任何作用的，因为最终输出的是embedding feature，和这两个一点关系都没有。如果把这两个范数留在loss中，那么在训练过程中，我们本来的训练目的是增大样本与类中心的余弦相似度，即增大$\cos \theta$值，但是，这两个范数的存在，可能会使神经网络不去增大余弦相似度，反而一直去增加这两个范数值，这就导致样本并没有在训练中去靠近类中心，同样也会影响类中心向量之间的离散化，没有达到训练目的。

解决措施就是，对这两个范数做归一化，归一化为常数，让网络在训练中只去优化余弦相似度。归一化就采用L2 Norm。feature norm之后，一般会加上一个常数scale，不加的话会使网络训练变得困难，因为会使网络过于关注难例。

logit形式：
$$
logit=s\cos \theta
$$
其中，s为scale超参数

loss形式：
$$
normface \ loss=-\frac{1}{N}\sum_i^N\log \frac{e^{s\cos \theta_{y_i}}}{e^{s\cos \theta_{y_i}}+\sum_{j\neq y_i}e^{s\cos \theta_j}} 
$$
其中，$y_i$表示样本i所属的类别

分类边界：
$$
\theta_1=\theta_2
$$

## angular margin

核心：压缩每一类样本的分布区域，使得类内距离更小，类间距离更大，使得模型能提取更discriminative的embedding。

三种形式：

1. A-softmax/SphereFace

   在样本与所属类别的夹角$\theta$乘上一个margin m，使$\theta$足够小才能把loss降下来

   正类logit形式：
   $$
   logit=s\cos (m\theta)
   $$
   loss形式：
   $$
   SphereFace\ loss=-\frac{1}{N}\sum_i^Nlog\frac{e^{scos(m\theta_{y_i})}}{e^{scos(m\theta_{y_i})}+\sum_{j\neq y_i}e^{scos(\theta_{j})}}
   $$
   分类边界：
   $$
   m\theta_1=\theta_2
   $$
   缺点：

   1. 单调性问题，$\theta \in[0,\pi]$，$\cos\theta$是单调减的，现在只有$\theta\in[0,\frac{\pi}{m}]$时，$\cos\theta$才是单调减的，若$\theta$不在这个范围内，那么就难于优化了。

      这个是有一个对应的解决措施：

      构造$\phi$函数，为多个减区间的$\cos$函数相加，整体为减函数，具体形式为
      $$
      \phi(\theta_j)=(-1)^ncos(m\theta_j)-2n,\ \ \theta_j\in[\frac{n\pi}{m},\frac{(n+1)\pi}{m}],\ \ n=0,1,..,m-1
      $$
      这样修正后，loss为
      $$
      SphereFace\ loss=-\frac{1}{N}\sum_i^Nlog\frac{e^{s\phi(\theta_{y_i})}}{e^{s\phi(\theta_{y_i})}+\sum_{j\neq y_i}e^{s\cos(\theta_{j})}}
      $$

   2. 两个类别之间的margin距离为$\frac{m-1}{m+1}\theta_{1,2}$，其中$\theta_{1,2}$为两个类中心$W_1,W_2$的夹角，那么这个实际的margin就会和两个类中心之间的夹角有关系，如果是两个非常接近的类，那么还是拉不开距离。

2. AM-Softmax/cosFace

   在样本与所属类别的余弦相似度$\cos$上减掉一个margin m，使余弦相似度足够大才能把loss降下来。

   正类logit形式：
   $$
   logit = s\cos \theta-m
   $$
   loss形式：
   $$
   cosFace\ loss=-\frac{1}{N}\sum_i^Nlog\frac{e^{scos\theta_{y_i}-m}}{e^{scos\theta_{y_i}-m}+\sum_{j\neq y_i}e^{scos\theta_{j}}}
   $$
   分类边界：
   $$
   \cos\theta_1-m=\cos\theta_2
   $$

3. ArcFace

   在样本与所属类别的夹角$\theta$上加上一个margin m，使夹角足够小才能把loss降下来。

   正类logit形式：
   $$
   logit=s\cos(\theta+m)
   $$
   loss形式：
   $$
   ArcFace\ loss=-\frac{1}{N}\sum_i^Nlog\frac{e^{scos(\theta_{y_i}+m)}}{e^{scos(\theta_{y_i}+m)}+\sum_{j\neq y_i}e^{scos(\theta_{j})}}
   $$
   分类边界：
   $$
   \theta_1+m=\theta_2
   $$
   单调性问题：

   $\theta+m$是可能大于$\pi$的，即可能离开减区间，解决方法：
   $$
   logit=\begin{cases}
   s\cos(\theta+m) & if \quad \cos \theta>\cos(\pi-m)\\
   s(\cos\theta-\sin(\pi-m)*m) & if\quad \cos \theta<\cos(\pi-m)
   \end{cases}
   $$
   可以发现，当$\theta+m>\pi$时，logit就是cosFace的形式。

4. 三者组合的combined face loss

   三种margin全部加上

   正类logit形式：
   $$
   logit=s\cos(m_1\theta+m_2)-m_3
   $$
   loss形式：
   $$
   combined\ face\ loss=-\frac{1}{N}\sum_i^Nlog\frac{e^{scos(m_1\theta_{y_i}+m_2)-m_3}}{e^{scos(m_1\theta_{y_i}+m_2)-m_3}+\sum_{j\neq y_i}e^{scos(\theta_{j})}}
   $$
   单调性问题：

   $m_1\theta+m_2$是可能大于$\pi$的，解决方法：（这个insightface和GluonFace里都没考虑，我觉得可以采用和SphereFace一样的方法）

   

   