import numpy as np
def min(*nums):
    """若nums各项为数字,则取所有数字的最小值;若为list,则逐元素取最小值
    """
    return np.min(nums,axis=0).tolist()

# def to_underline_naming(s):
    

# def to_hump_naming(s):
#     pass