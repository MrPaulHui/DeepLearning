# CNN中的OP

## Conv

### 前传

设输入的feature map的size为W*H\*C，卷积核size为K\*K，卷积核个数为N，pad为P，stride为S，则输出的feature map的size为
$$
out\ W=\frac{W+2P-K}{S}+1\\
out\ H=\frac{H+2P-K}{S}+1\\
out\ C=N
$$

### 反传

### 感受野

feature map上的一个像素**对应回原输入图的大小**

计算公式：

设第i卷积层的卷积核size为$k_i$，stride为$s _i$，输出的feature map为$FM_i$，$FM_i$的感受野为$RF_i$，累积stride为$s_{ti}=\prod_{j=1}^{i-1}s_j$，记$FM_0$为输入原图，$RF_0=1$

则第i+1层的感受野为
$$
RF_{i+1}=(k_{i+1}-1)s_{ti}+RF_{i}
$$
**推导过程**：

$FM_{i+1}$上的一个像素对应回$FM_i$上$k_{i+1}*k_{i+1}$个像素，而$FM_i$上一个像素点的感受野为$RF_i$，即$FM_i$上一个像素对应原图的$RF_i$个像素，$FM_i$上两个相邻的像素对应回原图的两个感受野之间跨度为$s_{ti}$（可以理解为第1到i层所有卷积层的作用等价于原图直接做了一次大小为$RF_i$，stride为$s_{ti}$的卷积，输出了$FM_i$），所以根据空间关系及参考卷积层输出公式，可得到公式。

所以可看出卷积层数越多，感受野越大，更大的stride可以带来更大的感受野，同理pooling层也会增大感受野。（关于pooling层的感受野，可以把pooling层看做特殊的卷积层，一样的公式）

## Pooling

降采样，带来一定范围内的平移不变性

## other Conv function

### 反卷积

参考：

https://blog.csdn.net/sinat_29957455/article/details/85558870

https://zhuanlan.zhihu.com/p/48501100

**反卷积并不是卷积的逆过程**，**本质也是卷积**，目的是**用卷积的方式增大feature map size**

一般的卷积操作，输出的feature map size是小于或等于输入的feature map size的，怎么使其大于呢？方法是**加更大的padding**，利用padding来增大输入feature map的实际size，再做卷积。

那么具体要加多少padding呢？需要根据公式
$$
o=\frac{i+2p-k}{s}+1
$$
需要注意，**反卷积的卷积核永远只以步幅s=1进行滑动**，那么设置stride意义是什么？**反卷积操作里设定的stride决定了padding的方式，若stride=1，则直接在feature map四周加padding，若stride>1，则需要在feature map原像素中加入间隔padding**。下面两幅图分别是stride=1和stride>1的情况。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190101215139260.gif)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190101220640484.gif)

根据设定的stride，**feature map原像素内部需要进行stride-1的间隔padding**

feature map内部已进行stride-1的间隔padding，所以，feature map size变为$(i-1)(stride-1)+i$，即为$(i-1)stride+1$，又实际滑动中的步幅都为1，所以根据公式，所需要填补在四周的padding为
$$
p=\frac{o+k-(i-1)stride-2}{2}
$$

### 3D卷积

参考：https://www.jianshu.com/p/09d1d8ffe8a4

设输入的多帧feature map size为$(N, C, D, H, W)$，其中N为batch size，C为每一帧feature map的channel数，H,W为每一帧feature map的长宽，D为帧数。

传统的2D卷积核格式为$(C_{out},C_{in},k\_height,k\_width)$

3D卷积核格式为$(C_{out},C_{in},k\_depth,k\_height,k\_width)$，多了帧数上的维度。每一帧上3D卷积和普通卷积是一样的操作，只是多了多帧同时卷积。看下面的代码大概可以明白

```python
a = torch.randn(1, 3, 10, 32, 32) #输入的feature map，C=3, D=10, H,W=32,32
c3d = torch.nn.Conv3d(3, 64, (5, 3, 3), padding=(0,1,1)) #3D卷积核
#out_C=64, k_depth=5, 意味着每5帧做一次3D卷积，5帧的结果汇聚到一帧输出的feature map上
out = c3d(a)
#out的shape为torch.Size([1, 64, 6, 32, 32]), 有6帧输出的feature map，因为帧数上没有pad，
#5帧一个单位滑动到最后，可以滑动6次，和空间上卷积是一个计算公式，至于CHW，则和普通空间卷积是一样的。
```

### 扩张卷积

参考：https://blog.csdn.net/gwplovekimi/article/details/90318426

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190816135631922.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly96aGFuZ3h1LmJsb2cuY3Nkbi5uZXQ=,size_16,color_FFFFFF,t_70)

如图，a为普通3\*3卷积；b为dilation=2的扩张卷积，实际size为5\*5；c为dilation=4的扩张卷积，实际size为9\*9

扩张后的卷积核大小与原卷积核大小的关系：
$$
new\_k=(dilation-1)(k-1)+k
$$
原理就是中间多出几个空洞，每两两原始卷积核元素之间增加（dilation-1）个空洞。

扩大了卷积核size，但又没增加参数量，因为扩张的部分都是0，只有原卷积核的参数有效。**增大了feature map的感受野。**

dilation=1即普通卷积。



