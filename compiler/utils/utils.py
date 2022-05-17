import numpy as np
def min(*nums):
    """若nums各项为数字,则取所有数字的最小值;若为list,则逐元素取最小值
    """
    return np.min(nums,axis=0).tolist()

# def to_underline_naming(s):
    

# def to_hump_naming(s):
#     pass

def select_no_sort(ls,num=1):
    if len(ls)<num:
        return
    if len(ls)==num:
        return ls
    size = len(ls)
    if num==1:
        for item in ls:
            yield item
    elif num==2 and size==3:
        i


if __name__=="__main__":
    for select in select_no_sort([0,1,2,3],3):
        print(select)
