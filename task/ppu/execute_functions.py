from executer.execute_functions import bind,bind_table
import os
import torch
from compiler.utils.utils import str_title
from collections import OrderedDict

save_data = True

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
    class_name_stats = []
    for op in self.op_list:
        class_name = str(type(op).__name__)
        class_name_stats.append(class_name)
        if class_name in bind_table:
            execute_fn = bind_table[class_name]
            execute_fn(op)
        else:
            assert False

    op_name_list = make_op_name(self.op_list)
    if save_data:
        tensor_stats = stats_tensors(op_name_list)
        inout_tensor_stats = stats_inout_tensors(op_name_list)
        save_to_file(direct="fwd",
                    op_stats=op_name_list,
                    tensor_stats=tensor_stats,
                    inout_tensor_stats=inout_tensor_stats,
                    ppu_instr=self.to_ppu_instr(),
                    path="test_data/forward")

@bind(operator="BackwardPPUFusedOp")
def backward_ppu_fused_op(self):
    class_name_stats = []
    for op in self.op_list:
        class_name = str(type(op).__name__)
        class_name_stats.append(class_name)
        if class_name in bind_table:
            execute_fn = bind_table[class_name]
            execute_fn(op)
        else:
            assert False
    op_name_list = make_op_name(self.op_list)
    if save_data:
        tensor_stats = stats_tensors(op_name_list)
        inout_tensor_stats = stats_inout_tensors(op_name_list)
        save_to_file(direct="bwd",
                    op_stats=op_name_list,
                    tensor_stats=tensor_stats,
                    inout_tensor_stats=inout_tensor_stats,
                    ppu_instr=self.to_ppu_instr(),
                    path="test_data/backward")

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


def stats_tensors(op_list):
    visit = set()
    record = OrderedDict()
    for op,op_name in op_list.items():
        op_tensor = {}

        for name,tensor in op.tensors.tensors.items():
            if tensor in visit:
                continue
            visit.add(tensor)
            if op_name.startswith("BackwardBatchnorm") and name=="mean":
                continue
            #BatchNorm的方差，转换成 1/sqrt(var+1e-5) ，因为S2Train上不方便算根号和倒数
            if name=="std":
                op_tensor[f"std_reci"] = 1/tensor.storage.data
                continue
            #BackwardScalarAdd用到的方差，转换为标准差
            if name=="bn_std":
                op_tensor[f"std"] = tensor.storage.data
                continue
            if not tensor.storage.data==None:
                op_tensor[name] = tensor.storage.data
        record[op_name] = op_tensor

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
        for name,tensor in op.tensors.tensors.items():
            if tensor in inout_tensors:
                if op_name.startswith("BackwardBatchnorm") and name=="mean":
                    continue
                #BatchNorm的方差，转换成 1/sqrt(var+1e-5) ，因为S2Train上不方便算根号和倒数
                if name=="std":
                    op_tensor[f"std_reci"] = 1/tensor.storage.data
                    continue
                #BackwardScalarAdd用到的方差，转换为标准差
                if name=="bn_std":
                    op_tensor[f"std"] = tensor.storage.data
                    continue
                if not tensor.storage.data==None:
                    op_tensor[name] = tensor.storage.data
        record[op_name] = op_tensor
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
    for op_name,tensor_list in inout_tensor_stats.items():
        for tensor_name,tensor in tensor_list.items():
            info.append(f"{op_name}.{tensor_name}.shape={list(tensor.shape)}")
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
    for op_name,tensor_list in tensor_stats.items():
        for tensor_name,tensor in tensor_list.items():
            info2.append(f"{op_name}.{tensor_name}.shape={tensor.shape}")
            tensor = tensor.reshape(-1).cpu().numpy()
            file_path = os.path.join(path,"detail",f"{op_name}.{tensor_name}.txt")
            if tensor.dtype==bool or tensor.dtype==np.int32:
                np.savetxt(file_path, tensor, fmt="%d", delimiter=" ")
            else:
                np.savetxt(file_path, tensor, fmt="%f", delimiter=" ")
        
    file_path = os.path.join(path,"detail",f"info.txt")
    with open(file_path,"w") as f:
        f.write("\n".join(info2))