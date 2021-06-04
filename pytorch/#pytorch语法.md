#pytorch语法

# pytorch基本语法

## tensor对象的创建

```python
import numpy as np
import torch
from torch import random
# tensor是一个任意维度的矩阵。但是一个张量中的所有元素的数据类型必须保持一致
# 这种类型可以在cpu上也可以在GPU上，可以采用dtype指定类型。可以用device指定设备：CPU or GPU
# torch.tensor
print("torch.Tensor 默认为；{}".format(torch.Tensor(1).dtype))
#torch.Tensor 默认为；torch.float32
#大写的tensor为浮点数
print("torch.tensor 默认为；{}".format(torch.tensor(1).dtype))
#torch.tensor 默认为；torch.int64
###########################
#同样可以用list构建tensor
a=torch.tensor([[1,2],[3,4]],dtype=torch.float64)
#也可以用ndarray创建，np中的向量底层是数组
b=torch.tensor(np.array([[1,2],[3,4]]),dtype=torch.uint8)
print(a,'\n',b)
#tensor([[1., 2.],
#        [3., 4.]], dtype=torch.float64) 
#tensor([[1, 2],
#       [3, 4]], dtype=torch.uint8)

#通过device指定设备,电脑没装cuda
# cuda0=torch.device('cuda:0')
# c=torch.ones((2,2),device=cuda0)
# print(c)
#几种常见的tensor创建方式
print(torch.arange(5),'\n')
#tensor([0, 1, 2, 3, 4]) 
print(torch.arange(1,5,2),'\n')
#tensor([1, 3])
print(torch.linspace(0,5,10),'\n') #start end padding
#tensor([0.0000, 0.5556, 1.1111, 1.6667, 2.2222, 2.7778, 3.3333, 3.8889, 4.4444,
#        5.0000]) 
print(torch.ones(3,3),'\n')
#tensor([[1., 1., 1.],
#        [1., 1., 1.],
#        [1., 1., 1.]]) 
print(torch.zeros(3,3),'\n')
#tensor([[0., 0., 0.],
#        [0., 0., 0.],
#        [0., 0., 0.]])
print(torch.rand(3,3),'\n')  #返回（0,1）均匀分布的矩阵
#tensor([[0.7020, 0.6953, 0.5130],
#       [0.8358, 0.4678, 0.9740],
#       [0.0502, 0.7943, 0.4515]]) 
print(torch.rand(3,3)*100,'\n')  #返回（0,。通过乘以100可以返回0-100
print('torch.randint(0,9,(3,3))\n',torch.randint(0,9,(3,3)))
print('torch.randn(3,3)，正态分布\n',torch.randn(3,3))

```

## tensor的统计函数，拼接，拆分

```python
import torch
import numpy as np
a=torch.rand(1,2,3,4,5) #生成一个5维的张量
# nums_element=a.nelement()
# nums_dimension=a.ndimension();
# nums_shape=a.size()
# print('\nnums_element=\n',nums_element,'\nnums_dimension=\n',nums_dimension,'\nnums_shape=\n',nums_shape)
# b=a.reshape(6,20)
# print(b.size())
# c=a.reshape(-1)   #flattem
# print(c.size())
# d=a.reshape(6,-1);
#squeeze 与 unsqueeze 用于去掉和添加轴
e=torch.squeeze(a)
print(e.shape)
e=torch.unsqueeze(e,0)
print(e.shape)
#可以用torch.t来转置矩阵
f=torch.tensor([[2,3]])
print(torch.t(f))
#矩阵拼接，cat和stack函数
#cat拼接要求行或者列一个轴的shape相同，而stack要求两个同型矩阵
g=torch.randn(2,3)
h=torch.randn(3,3)
cat_g_h=torch.cat((g,h))
print('\ncat_g_h\n',cat_g_h)
cat_g_g_g=torch.cat((g,g,g),dim=1) #dim=1按列拼接，dim=0，按行拼接
print(cat_g_g_g.shape)
#矩阵拆分
a=torch.randn(10,3)
for x in torch.split(a,[1,2,3,4],dim=0):   #拆分成迭代器
    print('\nx,shape\n',x.shape)
for x in torch.split(a,4,dim=0):           #按照4的大小拆分
    print('\nx.shape\n',x.shape)
for x in torch.chunk(a,4,dim=0):           #输出拆分矩阵个数
    print(x.shape)                

```

## pytorch的reduction操作

````python
#默认求取全局最大值
import torch
import numpy  as np
a=torch.tensor([[1,2],[3,4]])
print('全局最大值为{}'.format(torch.max(a)))
#指定维度dim后，返回最大值及其索引
print(torch.max(a,dim=1))  #求行或者列的最大值
# torch.return_types.max(
# values=tensor([2, 4]),
# indices=tensor([1, 1]))
print(torch.cumsum(a,dim=0))
# tensor([[1, 2],
#         [4, 6]])
print(torch.cumprod(a,dim=1)) 
# tensor([[ 1,  2],
#         [ 3, 12]])
#计算矩阵的均值，中值，协方差
a=torch.Tensor([[1,2],[3,4]])
#注意这个a是Tensor 浮点类型的
print(a.mean(),a.median(),a.std())
# tensor(2.5000) tensor(2.) tensor(1.2910)
a=torch.randint(0,3,(3,3))
print(torch.unique(a))
# tensor([0, 1, 2])
print(set(a))
# {tensor([1, 1, 1]), tensor([2, 0, 1]), tensor([2, 0, 0])}


````

## pytorch的自动微分

```python
import torch
#创建一个张量，设置 requires_grad=True 来跟踪与它相关的计算
x = torch.ones(2, 2, requires_grad=True)
print(x)
y=x+2
print(y)
#y作为操作的结果被创建，所以它有 grad_fn
print(y.grad_fn)

z=y*y*3
#计算z的算术平均值
out = z.mean()
print(z,out)
#后向传播，因为输出包含了一个标量，out.backward() 等同于out.backward(torch.tensor(1.))
out.backward()
##打印梯度  d(out)/dx 
print(x.grad)
```



