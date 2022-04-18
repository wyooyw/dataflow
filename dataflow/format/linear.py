import numpy as np
from format import *
'''
全连接层权重的数据格式转换
'''
def convert_weight(weight):
    assert weight.ndim==2
    output_length,input_length = weight.shape
    assert output_length%4==0
    weight = weight.reshape(output_length//4,4,input_length)
    weight = np.transpose(weight,(0,2,1))
    group,input_length,four = weight.shape
    linear_weight = LinearWeight()
    for group_idx in range(group):
        group_data = GroupData(data_list = [])
        for input_idx in range(input_length):
            group_data.add(Data(weight[group_idx,input_idx,:]))
        linear_weight_group = LinearWeightGroup( \
                                    group_number=GroupNumber(group_idx), \
                                    group_data=group_data)
        linear_weight.add(linear_weight_group=linear_weight_group)
    return linear_weight
    # return convert_input(weight)

'''
全连接层输入的数据格式转换
'''
def convert_input(input):
    assert input.ndim==2
    batch,length = input.shape
    assert batch%4==0
    assert length%4==0
    input = input.reshape(batch//4,4,length)
    input = np.transpose(input,(0,2,1))
    group, data_row_in_group, _ = input.shape
    
    linear_input = LinearInput()
    last_group_data = None
    for g in range(group):
        group_data = GroupData(data_list = [])
        # print(group_data)
        for row in range(data_row_in_group):
            group_number = g
            data_list = input[g,row,:]
            group_data.add(Data(data_list))
        linear_input_group = LinearInputGroup( \
                                group_number=GroupNumber(g), \
                                group_data=group_data)
        linear_input.add(linear_input_group=linear_input_group)
   
    return linear_input


if __name__=="__main__":
    input = np.arange(0,64).reshape(8,8)
    print(input)
    print(convert_input(input))

    # weight = np.arange(0,40).reshape(8,5)
    # print(weight)
    # print(convert_weight(weight))
    