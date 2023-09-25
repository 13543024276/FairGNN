#代码测试经验总结：
    #clone函数就是单射函数
    #detach就是剥离计算图，但是信息共享，修改同时修改，剥离后不参与任何计算图的跟踪



#代码测试AIP总结：

    #获取模型参数：
    # for name, parameters in model.named_parameters():
    #     # print(name)
    #     # print(parameters.data)
    #     tensor=torch.tensor(parameters.data)
    #     tensor.data[0][0][0][0]=1
    #     print(name, ':', torch.allclose(tensor,parameters.data))
    #     break















import numpy as np
# a=list([1,2,3])
# a=np.array(a)
# b=[True,True,True]
# b=np.array(b)
# a=a[b]
# print(a)

# labels=[1,0,1.1,1.5]
# print(labels)
# print(labels[labels>1])
# labels[labels>1]=1
# print(labels)

# str="sda"
# if str:
#     print(1)
# print(12)

import numpy as np
import torch
from scipy.sparse import csr_matrix
#
# # 用一个密集的矩阵作为参数
# A = np.array([
#     [1, 0, 2, 0],
#     [0, 0, 3, 0],
#     [4, 5, 6, 0]
# ])
# M1 = csr_matrix(A)
# print(M1)

#用一个（data，indices，indptr）元组作为参数
# data = np.array([1 ,2 ,3 ,4 ,5 ,6])
# indices = np.array([0 ,2 ,2 ,0 ,1 ,2])
# indptr = np.array([0 ,2 ,3 ,6])
# M5 = csr_matrix((data ,indices ,indptr), shape=(3 ,4))
# print(M5)
# def f(x):
#  return x*x
# A={j: i for i, j in enumerate([1,2,3,4])}
# L=list(map(A.get,[2,1]))
# print(L)


# import numpy as np
# import scipy.sparse as sp
# row = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
# col = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
#
# A = sp.coo_matrix(([0, 3, 2, 0, 0, 0, 7, 4, 0], (row, col)),
#                   shape=(3, 3), dtype=np.float32)
# print(A.toarray())
# print((A.T > A).toarray())
# print(A.T.multiply(A.T > A).toarray())
#
# A = A + A.T.multiply(A.T > A) - A.multiply(A.T > A)
# print("The symmetric adjacency matrix:", A.toarray(), sep='\n')


# a = np.array([2,4,6,8,10])
# #只有一个参数表示条件的时候
# print(a[np.where(a > 5)[0]])


#detach()和detach_()区别
#总结：克隆算子相当于自映射算子
# import  torch
# #
# a = torch.tensor([1, 2, 3.], requires_grad=False)
# b= torch.tensor([1, 2, 3.], requires_grad=True)
# m=b.clone()
# f=sum(m)
# d=f.detach()
# # m.requires_grad=True
# print("sasadasd")
# print(m)
# d.zero_()
# print(d)
# print(f)
# (f).sum().backward()
# print(m)
# print(b)
# print(a)
# c=a+b
#
# k=sum(a+c)
# k.backward()
# print(c.grad)#1,1,1
# print(a.grad)#2.2.2 None
# print(b.grad)#1.1.1
# # print(m.grad)#111
# print(b)
# # print(m)
# sum(m*2).backward()
# # sum(b+m).backward()
# print(b.grad)
# list1 = [1, 2, 3]
# es=list(list1)
# list1[2]=4
# print(list(list1)==es)
# print(list(list1))
# print(es)
# import torch.nn as nn
#



#这个就是测试如何获取模型的参数
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x
#
# model = Net()
#
# print(model)
# print()
#
# for name, parameters in model.named_parameters():
#     # print(name)
#     # print(parameters.data)
#     tensor=torch.tensor(parameters.data)
#     tensor.data[0][0][0][0]=1
#     print(name, ':', torch.allclose(tensor,parameters.data))
#     break


# import torch
#
# a = torch.tensor([1, 2, 3.], requires_grad=True)
# print(a.grad)
# out = a**2
# print(out)
#
# # 添加detach(),c的requires_grad为False
# c = out.detach()
# print(c)
# out.zero_()  # 使用in place函数对其进行修改
# print(c)
# # # 会发现c的修改同时会影响out的值
# # print(c)
# # print(out)
#
# # 这时候对c进行更改，所以会影响backward()，这时候就不能进行backward()，会报错
# out.sum().backward()
# print(a.grad)

#
# a = torch.tensor([1, 2, 3.], requires_grad=True)
# b= torch.tensor([1, 2, 3.], requires_grad=False)
#
# c=a+b
# m=c.clone()
# sum(m*5).backward()
# print(a.grad)
import  torch.nn as nn
import numpy as np
import torch
from tqdm import tqdm
class RanMatrix(nn.Module):
    def __init__(self,seed):
        super(RanMatrix, self).__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.randmatrix=torch.rand(4,4,requires_grad=True)
        self.random=nn.Parameter(self.randmatrix)
        print(f"一开始的randMatrix{self.randmatrix}")


    def forward(self, adj,k):
        mid = torch.sigmoid( k*self.random)
        return torch.mean(torch.square(mid - adj))


k=1
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

import scipy.sparse as sp

adj=torch.zeros(4,4)
adj[0][1]=0.5
adj[0][3]=0.5
adj[1][2]=0.6
adj[1][3]=0.6
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
# features = normalize(features)
adj = adj + torch.eye(adj.shape[0])
print(f"原来的矩阵：{adj}")
model=RanMatrix(seed)
optimizer_G = torch.optim.Adam(list(model.parameters()), lr=0.01)
for epoch in tqdm(range(1000)):
    optimizer_G.zero_grad()
    loss=model(adj,k)
    print(f"当前的损失值：{loss}")
    loss.backward()
    print(f"模型参数的梯度值：{model.random.grad}")
    optimizer_G.step()
with torch.no_grad():
    print(f"拟合的矩阵{torch.sigmoid(k*model.random)}")
    print(f"拟合的矩阵背后参数{model.random}")

for name, param in model.named_parameters():
    print(name, param.size())

