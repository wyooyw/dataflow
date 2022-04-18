import numpy as np
"""
换batch和channel两个轴
"""
def switch_batch_and_channel(tensor):
    return np.transpose(tensor,(1,0,2,3))

"""
转180度
"""
def reverse(tensor):
    return np.flip(tensor,[2,3]).copy()