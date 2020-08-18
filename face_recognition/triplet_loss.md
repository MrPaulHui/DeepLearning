# Triplet Loss

## 基本公式——三元组

- anchor 基准正例，记为a
- positive 和anchor同一类的正例，记为p
- negative 和negative不同类的负例，记为n

$$
triplet\_loss=\max(d(a,p)-d(a,n)+margin,0)
$$

其中，d表示距离

当$d(a,p)+margin<d(a,n)$，即类内距离足够比类间距离小的时候，认为当前的anchor样本已经训练的足够好了，所以不产生loss（为0）

## Triplet Mining

### triplet的难易程度

- easy triplet，满足$d(a,p)+margin<d(a,n)$的三元组，这种三元组不产生loss
- hard triplet，满足$d(a,n)<d(a,p)$的三元组，即类间距离比类内距离还小
- semi-hard triplet，满足$d(a,p)<d(a,n)<d(a,p)+margin$，即类间距离比类内距离大，但是不足够大（没有比类内距离加上余量还大）

对应的，每种三元组就对应每种negative example，即anchor和positive确定，哪些negative可以和anchor，positive构成easy/hard/semi-hard三元组。

<img src="https://pic3.zhimg.com/80/v2-7ec1af4273db2beaf6c43da3b7c087ce_720w.jpg" alt="img" style="zoom:50%;" />

上图中，红色部分表示hard negative examples，黄色部分表示semi-hard negative examples，绿色部分表示easy negative examples。

训练中，如果大量存在easy triplet，会严重降低训练效率（计算出三元组距离，却不产生loss，对训练没贡献）；如果都是hard triplet，则可能会导致训练难以收敛。最好的是用semi-hard，所以就需要进行triplet mining。

### online triplet mining

