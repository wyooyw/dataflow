import numpy as np
from bitarray import bitarray
import math
import torch
def min(*nums):
    """若nums各项为数字,则取所有数字的最小值;若为list,则逐元素取最小值
    """
    return np.min(nums,axis=0).tolist()

def int_to_bits(num,width,name="Num",endian="big"):
    assert num>=0 and num<(1<<width),f"{name} should in [{0},{(1<<width)-1}],but got {num}"
    bits = bitarray(endian=endian)
    bits.frombytes(int(num).to_bytes(math.ceil(width/8),byteorder=endian, signed=False))
    if endian=="big":
        bits = bits[-width:]
    else:
        bits = bits[0:width]
    return bits

# def to_underline_naming(s):
    

# def to_hump_naming(s):
#     pass

def select(ls,num):
    """从ls里选num个数
    """
    index_set = _select_index(len(ls),num)
    for index in index_set:
        yield [ls[idx] for idx in index]
        
def _select_index(n,m):
    """从0到n-1里选m个数，选出的组数内部，顺序都是从小到大。
    """
    assert n>0 and n<10 and m<=n and m >= 0
    array = list(range(0,n))
    result = []
    tmp = []
    def dfs(index,step):
        if step==m:
            result.append([*tmp])
            return
        for i in range(index,n):
            tmp.append(i)
            dfs(i+1,step+1)
            tmp.pop()
    dfs(0,0)
    return result



def padding_inside(tensor,padding=1):
    """在张量最后两维的内部做padding
    """
    if padding==0:
        return tensor
    batch,channel,height,width = tensor.shape
    padding_zeros = torch.zeros(batch,channel,height,width*padding)
    tensor = torch.cat( (tensor,padding_zeros),3 )
    tensor = tensor.reshape(batch,channel,height*(padding+1),width)

    tensor = torch.transpose(tensor,2,3)
    batch,channel,height,width = tensor.shape
    padding_zeros = torch.zeros(batch,channel,height,width*padding)
    tensor = torch.cat( (tensor,padding_zeros),3 )
    tensor = tensor.reshape(batch,channel,height*(padding+1),width)
    tensor = tensor[:,:,:-padding,:-padding]

    tensor = torch.transpose(tensor,2,3)
    return tensor


if __name__=="__main__":
    tensor = torch.range(1.0,9.0).reshape(1,1,3,3)
    print(padding_inside(tensor,1))