from executer.execute_functions import bind,bind_table
import os
import torch
from compiler.utils.utils import str_title
from collections import OrderedDict
from task.ppu.utils import convert_tensor_layout
import time
save_data = True
data_name = time.strftime('%Y-%m-%dT%H-%M-%S', time.localtime(time.time()))

@bind(operator="BackwardScalarAdd")
def backward_scalar_add(self):
    #prepare tensors
    output_grad = self.tensors.get_data("output_grad")
    output_grad_res = self.tensors.get_data("output_grad_res")
    bn_std = self.tensors.get_data("bn_std")

    #compute
    output_grad_res = torch.transpose(output_grad_res,1,3)
    input_grad = torch.mul(output_grad_res,bn_std)
    input_grad = torch.transpose(input_grad,1,3)
    input_grad = input_grad + output_grad
    
    #write back
    self.tensors.set_data("input_grad",input_grad)

@bind(operator="ForwardPPUFusedOp")
def forward_ppu_fused_op(self):

    # Execute each op in self.op_list
    for op in self.op_list:
        class_name = str(type(op).__name__)
        if class_name in bind_table:
            execute_fn = bind_table[class_name]
            execute_fn(op)
        else:
            assert False

    if save_data:
        op_name_list = make_op_name(self.op_list)
        tensor_stats = stats_tensors(op_name_list)
        inout_tensor_stats = stats_inout_tensors(op_name_list)
        save_to_file(direct="fwd",
                    op_stats=op_name_list,
                    tensor_stats=tensor_stats,
                    inout_tensor_stats=inout_tensor_stats,
                    ppu_instr=self.to_ppu_instr(),
                    path=f"test_data/{data_name}/forward")

@bind(operator="BackwardPPUFusedOp")
def backward_ppu_fused_op(self):
    for op in self.op_list:
        class_name = str(type(op).__name__)
        if class_name in bind_table:
            execute_fn = bind_table[class_name]
            execute_fn(op)
        else:
            assert False
    
    if save_data:
        op_name_list = make_op_name(self.op_list)
        tensor_stats = stats_tensors(op_name_list)
        inout_tensor_stats = stats_inout_tensors(op_name_list)
        save_to_file(direct="bwd",
                    op_stats=op_name_list,
                    tensor_stats=tensor_stats,
                    inout_tensor_stats=inout_tensor_stats,
                    ppu_instr=self.to_ppu_instr(),
                    path=f"test_data/{data_name}/backward")

@bind(operator="CrossEntropyLoss")
def cross_entropy_loss(self):
    torch_loss_func = torch.nn.CrossEntropyLoss()
    input = self.forward_softmax.tensors.get_data("input")
    input.requires_grad = True
    label = self.forward_entropy.tensors.get_data("label")

    loss = torch_loss_func(input,label)
    loss.backward()
    input_grad = input.grad
    self.backward_softmax.tensors.set_data("input_grad",input_grad)

op_name_reflect = {
    "ForwardAdd":"ForwardResAcc",
    "BackwardSplit":"BackwardResAccDouble",
    "BackwardScalarAdd":"BackwardResAccSingle"
}
def make_op_name(op_list):
    """为算子设置名称

    这里有的算子名称要替换名字（硬件同学熟悉的名字）
    遇到重复的算子，要在后面加上标号

    Return:
        {
            OP:name,
            OP:name,
            ...
        }
    """
    global op_name_reflect

    # 统计算子类名
    class_name_stats = {}
    class_name_index = {}
    op_name = OrderedDict()
    for op in op_list:
        class_name = type(op).__name__
        if class_name not in class_name_stats:
            class_name_stats[class_name] = 0
            class_name_index[class_name] = 1
        class_name_stats[class_name] += 1

    for op in op_list:
        # 原始类名
        class_name = type(op).__name__

        # 名字映射
        reflect_class_name = class_name
        if class_name in op_name_reflect:
            reflect_class_name = op_name_reflect[class_name]

        # 为名字编号
        if class_name_stats[class_name]>1:
            op_name[op] = f"{reflect_class_name}{class_name_index[class_name]}"
            class_name_index[class_name] += 1
        else:
            op_name[op] = reflect_class_name
    print(op_name.values(),len(op_list))
    return op_name

def mask_to_ptr(input,kernel_size):
    batch,channel,height,width = input.shape
    assert height%kernel_size==0 and width%kernel_size==0
    input = input.reshape(batch * channel * height//kernel_size,kernel_size,width//kernel_size,kernel_size)
    input = torch.transpose(input,1,2)
    _,indices = torch.nn.functional.max_pool2d(input, kernel_size,return_indices=True)
    indices = indices.reshape(batch,channel,height//kernel_size,width//kernel_size)
    return indices

def prepare_tensor(op,op_name,tensor,tensor_name):
    """ 对遍历到的tensor做一些修改

    1.BatchNorm的标准差,改为倒数
    2.BackwardScalarAdd的bn_std,改名为std
    3.BackwardBatchnorm的mean不需要(后续推一下一般的BN里是否需要这个数)
    4.判断数据是否为None
    5.特征图和特征图梯度均只要第0个Channel
    6.BatchNorm的mean和std只要第一个数
    7.relu的mask只要第一个channel
    """

    if tensor.storage.data==None:
        return None, None

    if op_name in ["ForwardMaxpool","BackwardMaxpool"] and tensor_name=="mask":
        tensor = mask_to_ptr(tensor.storage.data[:,0:1,:,:],op.attrs.get("kernel_size"))
        return "ptr", tensor[:,0:1,:,:]

    # if op_name.startswith("ForwardLinear") and tensor_name in ["input","output","input_grad","output_grad"]:
    #     print(tensor_name,tensor.shape,tensor.storage.data.shape)
    if tensor_name in ["input","output","input_grad","output_grad","mask","input2","output_grad2","output_grad_res"]:
        # print(op_name,tensor_name,tensor.shape,tensor.storage.data.shape)
        if tensor.storage.data.ndim==4:
            return tensor_name, tensor.storage.data[:,0:1,:,:]
        else:
            return tensor_name, tensor.storage.data

    if op_name.startswith("BackwardBatchnorm") and tensor_name=="mean":
        return None, None

    #BatchNorm的标准差，取倒数，这样S2Train上只需要做乘法
    if tensor_name=="std":
        return f"std_reci", 1/tensor.storage.data[0:1]

    #BackwardScalarAdd用到的标准差
    if tensor_name=="bn_std":
        return f"std", tensor.storage.data[0:1]

    if tensor_name=="mean":
        return tensor_name, tensor.storage.data[0:1]

    return tensor_name, tensor.storage.data

def layout_tensor(op,op_name,tensor_data,tensor_name):
    # print(type(tensor_data))
    if str(type(tensor_data))=="NoneType":
        return tensor_data

    # 所有张量按照PE输出的样子改一遍格式;maxpool的输出张量
    if op_name.startswith("ForwardMaxpool") and (tensor_name=="output" or tensor_name=="ptr"):
        kernel_size = op.attrs.get("kernel_size")
        tensor_data = convert_tensor_layout(tensor_data,div=kernel_size)
    elif op_name.startswith("BackwardMaxpool"):
        # 反传Maxpool，目前在resnet和alexnet，cifar10数据集下，其输入张量的长宽没有大于16的，所以这里其实什么也不做。
        # 但是后续需要考虑
        tensor_data = convert_tensor_layout(tensor_data)
    else:
        tensor_data = convert_tensor_layout(tensor_data)
    return tensor_data

def stats_tensors(op_list):
    visit = set()
    record = OrderedDict()
    for op,op_name in op_list.items():
        op_tensor = {}

        for tensor_name,tensor in op.tensors.tensors.items():
            if tensor in visit:
                continue
            visit.add(tensor)
            _tensor_name, _tensor = prepare_tensor(op, op_name, tensor, tensor_name)
            # _tensor = layout_tensor(op, op_name, _tensor, _tensor_name)
            if (not _tensor_name==None) and (not _tensor==None):
                op_tensor[_tensor_name] = _tensor
        record[op_name] = (op,op_tensor)

    return record

def stats_inout_tensors(op_list):
    
    include = set() #所有涉及到的张量
    exclude = set()
    include_name = set()
    exclude_name = set()
    for op,_ in op_list.items():
        for name,tensor in op.tensors.tensors.items():
            if tensor in include:
                exclude.add(tensor)
                exclude_name.add(name)
            else:
                include.add(tensor)
                include_name.add(name)

    inout_tensors = include - exclude

    record = {}
    for op,op_name in op_list.items():
        op_tensor = {}
        for tensor_name,tensor in op.tensors.tensors.items():
            if tensor in inout_tensors:
                _tensor_name, _tensor = prepare_tensor(op, op_name, tensor, tensor_name)
                # _tensor = layout_tensor(op, op_name, _tensor, _tensor_name)
                if (not _tensor_name==None) and (not _tensor==None):
                    op_tensor[_tensor_name] = _tensor
        record[op_name] = (op,op_tensor)
    return record

import numpy as np
index_fwd = 0
index_bwd = 0
def save_to_file(direct,op_stats,tensor_stats,inout_tensor_stats,ppu_instr,path="test_data"):
    """保存到文件
    """
    global index_fwd
    global index_bwd
    if direct=="fwd":
        path = os.path.join(path,str(index_fwd))
        os.makedirs(path,exist_ok=True) 
        os.makedirs(os.path.join(path,"detail"),exist_ok=True) 
        index_fwd += 1
    elif direct=="bwd":
        path = os.path.join(path,str(index_bwd))
        os.makedirs(path,exist_ok=True) 
        os.makedirs(os.path.join(path,"detail"),exist_ok=True) 
        index_bwd += 1

    
    info = []
    info.append(str_title("Operators"))
    for op_name in tensor_stats.keys():
        info.append(op_name)
    info.append("\n")
    info2 = [*info]

    # PPU instr
    ppu_instr_filepath = os.path.join(path,"config.txt")
    with open(ppu_instr_filepath,"w") as f:
        f.write(ppu_instr.export().to01())

    # Save input & output tensors
    info.append(str_title("Tensors"))
    for op_name,value in inout_tensor_stats.items():
        op,tensor_list = value
        for tensor_name,tensor in tensor_list.items():
            info.append(f"{op_name}.{tensor_name}.shape={list(tensor.shape)}")
            tensor = layout_tensor(op,op_name,tensor,tensor_name)
            tensor = tensor.reshape(-1).cpu().numpy()
            file_path = os.path.join(path,f"{op_name}.{tensor_name}.txt")
            if tensor.dtype==bool or tensor.dtype==np.int32:
                np.savetxt(file_path, tensor, fmt="%d", delimiter=" ")
            else:
                np.savetxt(file_path, tensor, fmt="%.8f", delimiter=" ")
        
    file_path = os.path.join(path,f"info.txt")
    with open(file_path,"w") as f:
        f.write("\n".join(info))
        f.write("\n")
        f.write(str(ppu_instr))

    # Save all tensors in 'detail' folder
    info2.append(str_title("Tensors"))
    for op_name,value in tensor_stats.items():
        op,tensor_list = value
        for tensor_name,tensor in tensor_list.items():
            info2.append(f"{op_name}.{tensor_name}.shape={tensor.shape}")
            tensor = layout_tensor(op,op_name,tensor,tensor_name)
            tensor = tensor.reshape(-1).cpu().numpy()
            file_path = os.path.join(path,"detail",f"{op_name}.{tensor_name}.txt")
            if tensor.dtype==bool or tensor.dtype==np.int32:
                np.savetxt(file_path, tensor, fmt="%d", delimiter=" ")
            else:
                np.savetxt(file_path, tensor, fmt="%f", delimiter=" ")
        
    file_path = os.path.join(path,"detail",f"info.txt")
    with open(file_path,"w") as f:
        f.write("\n".join(info2))