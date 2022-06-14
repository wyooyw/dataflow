import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from compiler.target_gen.memory.memory_manager import MemoryManager
from compiler.scheduler.normal_scheduler import NormalScheduler
from compiler.scheduler.wu_imm_scheduler import WUImmScheduler
from simulator.memory import Memory
import torch.nn as nn
from model.resnet import resnet18_cifar
from model.lenet import *
from model.alexnet import AlexNet

from compiler.utils.unique_class_name import unique_class_name
from compiler.graph.replace_tool import Finder,ReplaceTool
from compiler.config import Config,CodeGen
from converter import Converter

from backends.sparse_train.target_code.instruction_gen import InstructionGenerator

from compiler.target.dataflow import Dataflow
import numpy as np
def run():
    torch_net = TestNet()
    # torch_net = resnet18_cifar()
    # torch_net = AlexNet()
    # net = torchvision.

    total_params = sum(p.numel() for p in torch_net.parameters())
    in_shape=[4,3,32,32]
    print("in_shape:",in_shape)
    converter = Converter(torch_net,in_shape=in_shape)
    converter.convert()
    net = converter.net
    replace_tool = ReplaceTool(net=net,config_path="./backends/sparse_train/replace.yaml")
    # replace_tool.replace_all()
    # scheduler = NormalScheduler()
    scheduler = WUImmScheduler()
    scheduler.schedule(net)
    print(net)
    print(net.count())
    net.reduce_tensor()
    print(net.count())
    net.set_tensor_index()

    MemoryManager().tensor_memory_layout2(net)
    from functools import reduce
    input = torch.randn(in_shape)
    input.requires_grad=True

    output = Memory().get(net.sim_run_to(input,"BEdge_0").tensors.get("output_grad").addr)
    torch_output = torch_net(input)
    torch_output = torch.sum(torch_output)
    torch_output.backward()
    torch_output = input.grad
    print("my   :",output)
    print("torch:",torch_output)
    if output.shape==torch_output.shape:
        print(torch.max(torch.abs(output-torch_output))<0.01)
    else:
        print(f"Shape is not equal! output.shape={output.shape}, torch_output.shape={torch_output.shape}")
    
    
if __name__=="__main__":
    run()