import numpy as np
from bitarray import bitarray
import math
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

# def select_no_sort(ls,num=1):
#     if len(ls)<num:
#         return
#     if len(ls)==num:
#         return ls
#     size = len(ls)
#     if num==1:
#         for item in ls:
#             yield item
#     elif num==2 and size==3:
#         i


# if __name__=="__main__":
#     for select in select_no_sort([0,1,2,3],3):
#         print(select)
