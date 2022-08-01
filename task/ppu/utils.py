import torch.nn as nn
import torch
from task.ppu.op.scalar_add import BackwardScalarAdd

def judge_single_line(operator):
    if not type(operator).__name__ in ["BackwardSplit"]:
        return
    predecessors = operator.predecessor
    assert len(predecessors)==2
    first,second = predecessors
    first_is_conv = type(first).__name__=="BackwardConv"
    second_is_conv = type(second).__name__=="BackwardConv"
    if first_is_conv and second_is_conv:
        # operator._mul_std = False
        return False
    elif first_is_conv or second_is_conv:
        _main_add_tensor = second.tensors.output
        last_split = first
        if first_is_conv:
            _main_add_tensor = first.tensors.output
            last_split = second
        operator._main_add_tensor = _main_add_tensor
        operator._mul_std = True
        
        # 拿到BN的std
        bn = (last_split.successor - set([operator]))[0]
        operator._bn_std = bn.tensors.get("std")
        operator._bn_input_grad = bn.tensors.get("input_grad")

        print(operator.name,"judge_single_line")
        return True
    else:
        assert False,f"Two predecessors are {first.name} and {second.name}"

def replace(net):
    op_list = []
    for op in net.topo():
        if judge_single_line(op):
            op_list.append(op)
    print([op.name for op in op_list])
    for op in op_list:
        replace_op = BackwardScalarAdd.replace_from(op)
        
        # 重连后继节点
        successors = set(op.successor)
        for successor in successors:
            successor.disconnect_predecessor(op)
            successor.connect_predecessor(replace_op)

        # 重连前驱节点
        predecessors = set(op.predecessor)
        for predecessor in predecessors:
            predecessor.disconnect_successor(op)
            predecessor.connect_successor(replace_op)
        
        #替换网络中的算子
        net.remove_operator(op)
        net.add_operator(replace_op)

def remove_bn_affine(net):
    def _remove_bn_affine(module):
        pass
    #     if type(module)==nn.BatchNorm2d:
    #         num_features = module.num_features
    #         weight = module.weight.detach()
    #         bias = module.bias.detach()
    #         if module.weight==None and module.bias==None:
    #             return
    #         mean = module.running_mean.detach()
    #         var = module.running_var.detach()
    #         new_var = var/torch.square(weight)
    #         new_mean = mean - torch.mul(bias,new_var)

    #         module.affine = False
    #         module.running_mean = new_mean
    #         module.running_var = new_var
    #         module.weight = None
    #         module.bias = None
    # net.apply(_remove_bn_affine)


def _convert_tensor_layout_4d(tensor,out_tile_len):
    batch,channel,height,width = tensor.shape
    assert width%out_tile_len==0 or width<=out_tile_len,"Width of the tensor is not good."
    if width > out_tile_len:
        tensor = tensor.reshape(batch,channel,height,width//out_tile_len,out_tile_len)
        tensor = tensor.transpose(2,3)
    return tensor

def _convert_tensor_layout_2d(tensor,out_tile_len):
    return tensor
    
def convert_tensor_layout(tensor,out_tile_len=16,div=1):
    out_tile_len = out_tile_len // div
    if tensor.ndim==4:
        return _convert_tensor_layout_4d(tensor,out_tile_len)
    elif tensor.ndim==2:
        return _convert_tensor_layout_2d(tensor,out_tile_len)
    else:
        return tensor
        # assert False,f"tensor.ndim={tensor.ndim}"

if __name__=="__main__":
    tensor = torch.arange(0,64).reshape(1,1,8,8)
    print(tensor)
    tensor_layout = _convert_tensor_layout_4d(tensor,out_tile_len=10)
    print(tensor_layout)