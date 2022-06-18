import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


from compiler.target_gen.memory.memory_manager import MemoryManager
from compiler.graph_ir.operators.batchnorm import ForwardBatchnorm,BackwardBatchnorm
from compiler.scheduler.normal_scheduler import NormalScheduler
from compiler.scheduler.wu_imm_scheduler import WUImmScheduler
from simulator.memory import Memory
import torch.nn as nn
from model import resnet18_cifar,AlexNet
import model as Model
from compiler.graph.replace_tool import ReplaceTool
from converter import Converter
from compiler.target_gen.memory.storage import StorageType

from compiler.target.dataflow import Dataflow
import numpy as np
def run():
    # torch_net = Model.TestNet()
    torch_net = Model.resnet18_cifar()
    # torch_net = Model.AlexNet()
    # net = torchvision.

    total_params = sum(p.numel() for p in torch_net.parameters())
    in_shape=[4,3,32,32]
    print("in_shape:",in_shape)
    converter = Converter(torch_net,in_shape=in_shape)
    # print(converter.trace.graph)
    converter.convert()
    net = converter.net
    
    #将BN算子的avg,std,alpha,beta合并成一个tensor
    #编译器里不好实现，这里hack一下
    tmp = {}
    for op in net.topo():
        if type(op)==ForwardBatchnorm or type(op)==BackwardBatchnorm:
            avg_storage = op.tensors.get("avg").storage
            if avg_storage in tmp:
                bn_use = tmp[avg_storage]
            else:
                bn_use = MemoryManager().allocWeight((4,op.tensors.get("avg").shape[0]))
                avg_data = op.tensors.get("avg").storage.data
                std_data = op.tensors.get("std").storage.data
                alpha_data = op.tensors.get("alpha").storage.data
                beta_data = op.tensors.get("beta").storage.data
                bn_use_data = np.vstack((avg_data,std_data,alpha_data,beta_data))
                bn_use.storage.data = bn_use_data
                tmp[avg_storage] = bn_use
            op.tensors.set("bn_use",bn_use)
            op.tensors.add_read_tensor("bn_use")
            op.tensors.tensors.pop("avg")
            op.tensors.tensors.pop("std")
            op.tensors.tensors.pop("alpha")
            op.tensors.tensors.pop("beta")
    print("======================== Net =========================")
    print(net)

    print("=================== Net after operator merge ====================")
    replace_tool = ReplaceTool(net=net,config_path="./backends/sparse_train/replace.yaml")
    replace_tool.replace_all()
    print(net)
    

    print("=================== Net after schedule ====================")
    # scheduler = NormalScheduler()
    scheduler = WUImmScheduler()
    scheduler.schedule(net)
    print(net)
    print("\n======================== Memory =========================")
    storage_record = {
        StorageType.ACTIVATION:[],
        StorageType.FEATURE_GRAD:[],
        StorageType.WEIGHT_GRAD:[],
        StorageType.WEIGHT:[]
    }
    storage_stats = {
        StorageType.ACTIVATION:0,
        StorageType.FEATURE_GRAD:0,
        StorageType.WEIGHT_GRAD:0,
        StorageType.WEIGHT:0
    }
    storage_visit = {}
    for op in net.topo():
        print(f"{op.name}:")
        for key,tensor in op.get_tensors().tensors.items():
            if tensor:
                storage = tensor.storage
                if storage not in storage_visit:
                    storage_visit[storage] = f"{op.name}.{key}"
                    print(f"  [{storage.type}] {key} shape={tensor.shape} size={storage.size}")

                    storage_record[storage.type].append(storage)
                    storage_stats[storage.type] += storage.size
                else:
                    # pass
                    print(f"  [{storage.type}] {key} (share storage with {storage_visit[storage]})")
            else:
                # pass
                print(f"  [None] {key}")
    total = 0
    for key,stats in storage_stats.items():
        total += stats
        b = stats*2
        kb = b/1024
        mb = kb/1024
        b = int(b*100)/100
        kb = int(kb*100)/100
        mb = int(mb*100)/100
        print(f"{key}:{stats} num, {b}B = {kb}KB = {mb}MB")
    # print("total num:",total)
    print(storage_stats)
    
    print(net.count())
    net.reduce_tensor()
    print(net.count())
    net.set_tensor_index()

    # MemoryManager().tensor_memory_layout2(net,show_image=True,save_image=True)
    MemoryManager().tensor_memory_layout2(net,show_image=True)
    dataflow = Dataflow(net)
    dataflow.generate()
    # from functools import reduce
    # # input = torch.range(1.0,reduce(lambda x,y:x*y,in_shape)).reshape(in_shape)
    # input = torch.randn(in_shape)
    # input.requires_grad=True

    # output = Memory().get(net.sim_run_to(input,"BBatchnorm_0").tensors.get("input_grad").addr)
    # torch_output = torch_net(input)
    # torch_output = torch.sum(torch_output)
    # torch_output.backward()
    # torch_output = input.grad
    # print("my   :",output)
    # print("torch:",torch_output)
    # if output.shape==torch_output.shape:
    #     print(torch.max(torch.abs(output-torch_output))<0.01)
    # else:
    #     print(f"Shape is not equal! output.shape={output.shape}, torch_output.shape={torch_output.shape}")
        
    MemoryManager().count_read_and_write_times(net)
    # print(net.hash["BConv_1"].get_tensors().tensors["weight"].storage.data)
    # print(net.hash["BConv_0"].get_tensors().tensors["weight"].storage.data)
    
if __name__=="__main__":
    run()