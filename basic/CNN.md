# CNN中的OP

## Conv

### 前传

设输入的feature map的size为W*H\*C，卷积核size为K\*K，卷积核个数为N，pad为P，stride为S，则输出的feature map的size为
$$
out\ W=\frac{W+P-2K}{S}+1\\
out\ H=\frac{H+P-2K}{S}+1\\
out\ C=N
$$

### 反传



## Pooling

