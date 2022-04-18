import numpy as np
from test_op.format.format import *

'''
卷积层的输入的格式转换
'''
def convert_input(input,core_width=3,tile_width=4):
    assert input.ndim==4
    batch,channel,height,width = input.shape
    assert height%4==0
    assert batch==1
    input = input.reshape(channel,height//4,4,width)
    input = np.transpose(input,(1,3,0,2)) #height//4,width,channel,4
    
    group_col, width,channel,_ = input.shape
    width_step = tile_width - core_width + 1
    width_begin = 0
    conv_input = ConvInput()
    group_row_idx = 0
    while width_begin + width_step < width:
        for group_col_idx in range(group_col):
            group_data = GroupData(data_list = [])
            for width_idx in range(tile_width):
                for channel_idx in range(channel):
                    input_group = input[group_col_idx,width_begin+width_idx,channel_idx,:]
                    group_data.add(Data(input_group))
            conv_input_group = ConvInputGroup( \
                                    group_number=GroupNumber(group_col * group_row_idx + group_col_idx), \
                                    group_data=group_data)
            conv_input.add(conv_input_group=conv_input_group)

        width_begin += width_step
        group_row_idx += 1

    return conv_input


'''
卷积层权重的格式转换
输入：[output_channel, input_channel, core_height, core_width]
输出：[output_channel//(4*PE_ARRAY_NUM*CORE_NUM), CORE_NUM, PE_ARRAY_NUM,  core_height, input_channel* core_width, 4]
output_channel//(4*PE_ARRAY_NUM*CORE_NUM) > 1时，为卷积核没法一次加载到芯片上，需要分批次加载
'''
def convert_weight(conv_weight,CORE_NUM=1,PE_ARRAY_NUM=1):
    assert conv_weight.ndim==4
    assert conv_weight.shape[0]%(4*PE_ARRAY_NUM*CORE_NUM)==0
    output_channel, input_channel, core_height, core_width = conv_weight.shape
    conv_weight = conv_weight.reshape(output_channel//(4*PE_ARRAY_NUM*CORE_NUM), CORE_NUM, PE_ARRAY_NUM, 4, input_channel, core_height, core_width)
    
    #换轴，变成：[output_channel//(4*PE_ARRAY_NUM*CORE_NUM), CORE_NUM, PE_ARRAY_NUM, core_height, input_channel, core_width, 4]
    conv_weight = np.transpose(conv_weight,(0,1,2,5,4,6,3))

    #将input_channel, core_width两个轴合并（因为这里面的数据都是一组的）
    conv_weight = conv_weight.reshape(output_channel//(4*PE_ARRAY_NUM*CORE_NUM), CORE_NUM, PE_ARRAY_NUM, core_height, input_channel* core_width, 4)
    
    #开始生成bits
    layer_num,core_num,pe_array_num,core_height,in_group_index,four = conv_weight.shape
    f_conv_weight = ConvWeight()
    for i_core in range(core_num):
        for i_pe_array in range(pe_array_num):
            for i_group in range(core_height):
                f_group_number = GroupNumber(group_number=i_group, \
                                    prefix = Prefix(core_idx=i_core, pe_array_idx=i_pe_array))
                f_group_data = GroupData(data_list = [])
                group_data = conv_weight[0,i_core,i_pe_array,i_group,...]
                for i_index in range(in_group_index):
                    f_group_data.add(Data(group_data[i_index,...]))

                f_conv_weight_group = ConvWeightGroup( \
                                    group_number=f_group_number, \
                                    group_data=f_group_data)
                f_conv_weight.add(conv_weight_group=f_conv_weight_group)
    return f_conv_weight

if __name__=="__main__":
    weight = np.arange(0,72).reshape(4,2,3,3)
    format_weight = convert_weight(weight,PE_ARRAY_NUM=1,CORE_NUM=1)
    print(format_weight)
    # with open(f"conv_input.bits",'wb') as f:
    #     format_input.to_bits().tofile(f)